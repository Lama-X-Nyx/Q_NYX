"""
Dataset Manifest Generator

Creates reproducible manifests of validation datasets.
Ensures exact reproducibility of validation runs.

Usage:
    python -m src.validation.manifest --data-dir data/raw --output data/manifests/validation_manifest.json
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def hash_file(filepath: Path) -> str:
    """Calculate SHA256 hash of file"""
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def analyze_csv(filepath: Path) -> Dict:
    """Analyze CSV file and extract metadata"""
    
    df = pd.read_csv(filepath)
    
    # Detect datetime column
    datetime_col = None
    for col in ['datetime', 'timestamp', 'date', 'time']:
        if col in df.columns:
            datetime_col = col
            break
    
    metadata = {
        'rows': len(df),
        'columns': list(df.columns),
        'size_bytes': filepath.stat().st_size
    }
    
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        metadata['start_date'] = df[datetime_col].min().isoformat()
        metadata['end_date'] = df[datetime_col].max().isoformat()
        metadata['datetime_column'] = datetime_col
    
    return metadata


def create_manifest(data_dir: str, pairs: Optional[List[str]] = None) -> Dict:
    """
    Create manifest for validation datasets
    
    Args:
        data_dir: Directory containing OHLCV CSV files
        pairs: List of pairs to include (default: all)
    
    Returns:
        Manifest dictionary
    """
    
    pairs = pairs or []

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    manifest = {
        'created_at': datetime.now().isoformat(),
        'data_directory': str(data_path),
        'files': {}
    }
    
    # Find CSV files
    csv_files = list(data_path.glob('*.csv'))
    
    if pairs:
        # Filter by pairs
        csv_files = [f for f in csv_files if any(pair in f.name for pair in pairs)]
    
    for filepath in sorted(csv_files):
        file_info = {
            'path': str(filepath),
            'filename': filepath.name,
            'hash_sha256': hash_file(filepath),
            'modified_at': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
        }
        
        # Analyze CSV content
        try:
            csv_metadata = analyze_csv(filepath)
            file_info.update(csv_metadata)
        except Exception as e:
            file_info['error'] = str(e)
        
        manifest['files'][filepath.name] = file_info
    
    # Summary
    manifest['summary'] = {
        'total_files': len(manifest['files']),
        'total_rows': sum(f.get('rows', 0) for f in manifest['files'].values()),
        'pairs_included': list(manifest['files'].keys())
    }
    
    return manifest


def save_manifest(manifest: Dict, output_path: str):
    """Save manifest to JSON file"""
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Manifest saved: {output}")
    print(f"  Files: {manifest['summary']['total_files']}")
    print(f"  Total rows: {manifest['summary']['total_rows']:,}")


def load_manifest(manifest_path: str) -> Dict:
    """Load manifest from JSON file"""
    
    with open(manifest_path, 'r') as f:
        return json.load(f)


def verify_manifest(manifest_path: str, data_dir: str) -> bool:
    """
    Verify that current data matches manifest
    
    Returns:
        True if all files match, False otherwise
    """
    
    manifest = load_manifest(manifest_path)
    data_path = Path(data_dir)
    
    all_match = True
    
    for filename, file_info in manifest['files'].items():
        filepath = data_path / filename
        
        if not filepath.exists():
            print(f"✗ Missing file: {filename}")
            all_match = False
            continue
        
        current_hash = hash_file(filepath)
        expected_hash = file_info['hash_sha256']
        
        if current_hash != expected_hash:
            print(f"✗ Hash mismatch: {filename}")
            print(f"  Expected: {expected_hash[:16]}...")
            print(f"  Got:      {current_hash[:16]}...")
            all_match = False
        else:
            print(f"✓ Verified: {filename}")
    
    return all_match


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate validation dataset manifest')
    parser.add_argument('--data-dir', default='data/raw', help='Data directory')
    parser.add_argument('--output', default='data/manifests/validation_manifest.json', help='Output manifest file')
    parser.add_argument('--pairs', nargs='+', help='Pairs to include (default: all)')
    parser.add_argument('--verify', action='store_true', help='Verify existing manifest')
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify mode
        print(f"Verifying manifest: {args.output}")
        if verify_manifest(args.output, args.data_dir):
            print("\n✓ All files verified")
        else:
            print("\n✗ Verification failed")
    else:
        # Create mode
        print(f"Creating manifest from: {args.data_dir}")
        manifest = create_manifest(args.data_dir, args.pairs)
        save_manifest(manifest, args.output)
