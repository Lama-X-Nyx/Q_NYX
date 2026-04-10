import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const App = () => {
  const [selectedPair, setSelectedPair] = useState('BTCUSDT');
  const [pairs, setPairs] = useState([]);
  const [results, setResults] = useState(null);
  const [livePrice, setLivePrice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [historicalData, setHistoricalData] = useState([]);
  const [params, setParams] = useState({ sdc: 3.5, prob: 0.55, pyramid: 0.10, stop: 0.05 });

  useEffect(() => {
    fetchPairs();
  }, []);

  useEffect(() => {
    if (selectedPair) {
      fetchLivePrice();
      fetchHistoricalData();
      const interval = setInterval(fetchLivePrice, 5000);
      return () => clearInterval(interval);
    }
  }, [selectedPair]);

  const fetchPairs = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/pairs`);
      setPairs(res.data.pairs);
      if (res.data.pairs.length > 0 && !selectedPair) {
        setSelectedPair(res.data.pairs[0]);
      }
    } catch (error) {
      console.error('Error fetching pairs:', error);
    }
  };

  const fetchLivePrice = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/live/price/${selectedPair}`);
      setLivePrice(res.data);
    } catch (error) {
      console.error('Error fetching price:', error);
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/data/${selectedPair}?limit=100`);
      setHistoricalData(res.data.data || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/api/backtest/${selectedPair}`);
      setResults(res.data);
    } catch (error) {
      console.error('Error running backtest:', error);
      alert('Error running backtest: ' + error.message);
    }
    setLoading(false);
  };

  const COLORS = ['#10b981', '#ef4444'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-4 md:p-6">
      <div className="max-w-[1800px] mx-auto">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 mb-6 shadow-2xl">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
                🚀 NYX ULTIMATE
              </h1>
              <p className="text-sm opacity-80 mt-1">Professional Trading System • Live Analytics • Multi-Asset</p>
            </div>
          </div>
          
          {livePrice && (
            <div className="mt-4 pt-4 border-t border-white/20 flex gap-6 text-sm flex-wrap">
              <div><span className="opacity-70">Price:</span> <span className="font-bold text-lg">${livePrice.price.toFixed(2)}</span></div>
              <div><span className="opacity-70">24h Change:</span> <span className={`font-bold ${livePrice.change_24h > 0 ? 'text-green-400' : 'text-red-400'}`}>{livePrice.change_24h > 0 ? '+' : ''}{livePrice.change_24h.toFixed(2)}%</span></div>
              <div><span className="opacity-70">Volume:</span> <span className="font-bold">${(livePrice.volume_24h / 1000000).toFixed(2)}M</span></div>
              <div><span className="opacity-70">High:</span> <span className="font-bold">${livePrice.high_24h.toFixed(2)}</span></div>
              <div><span className="opacity-70">Low:</span> <span className="font-bold">${livePrice.low_24h.toFixed(2)}</span></div>
            </div>
          )}
        </div>
        
        {/* Tabs */}
        <div className="bg-white/10 border border-white/20 rounded-xl p-2 mb-6 flex gap-2 overflow-x-auto">
          {['overview', 'backtest', 'charts', 'parameters'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg whitespace-nowrap font-semibold transition ${
                activeTab === tab ? 'bg-gradient-to-r from-purple-500 to-pink-500' : 'hover:bg-white/10'
              }`}
            >
              {tab === 'overview' && '📊'} {tab === 'backtest' && '⚙️'} {tab === 'charts' && '📈'} {tab === 'parameters' && '🎛️'} {tab.toUpperCase()}
            </button>
          ))}
        </div>
        
        {/* Pair Selector */}
        <div className="flex gap-3 mb-6 overflow-x-auto">
          {pairs.map(pair => (
            <button
              key={pair}
              onClick={() => setSelectedPair(pair)}
              className={`px-6 py-3 rounded-xl font-semibold transition ${
                selectedPair === pair
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 shadow-lg scale-105'
                  : 'bg-white/10 border border-white/20 hover:scale-105'
              }`}
            >
              {pair}
            </button>
          ))}
        </div>
        
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <>
            <button
              onClick={runBacktest}
              disabled={loading}
              className="w-full bg-gradient-to-r from-green-500 to-emerald-500 py-4 rounded-xl font-bold text-lg mb-6 hover:opacity-90 transition disabled:opacity-50"
            >
              {loading ? '⏳ Running Backtest...' : '🚀 Run Backtest'}
            </button>

            {results && !results.error && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <MetricCard label="Total Return" value={`+${results.total_return.toFixed(2)}%`} sub={`$${results.final_capital.toLocaleString()}`} />
                  <MetricCard label="CAGR" value={`${results.cagr.toFixed(2)}%`} sub="Annual" />
                  <MetricCard label="Win Rate" value={`${results.win_rate.toFixed(1)}%`} sub={`${results.winners}/${results.total_trades}`} />
                  <MetricCard label="Profit Factor" value={results.profit_factor.toFixed(2)} sub="R/R" />
                  <MetricCard label="Total Trades" value={results.total_trades} sub={`${(results.total_trades/results.period_years/12).toFixed(0)}/mo`} />
                  <MetricCard label="Winners" value={results.winners} sub="Profitable" />
                  <MetricCard label="Avg Win" value={`+${results.avg_win.toFixed(2)}%`} sub="Per win" />
                  <MetricCard label="Avg Loss" value={`-${results.avg_loss.toFixed(2)}%`} sub="Per loss" />
                </div>

                <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-xl p-6">
                  <h3 className="text-xl font-bold mb-4">💼 Recent Trades</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/20">
                          <th className="text-left p-2">Date</th>
                          <th className="text-left p-2">Type</th>
                          <th className="text-right p-2">Entry</th>
                          <th className="text-right p-2">Exit</th>
                          <th className="text-right p-2">P&L</th>
                          <th className="text-right p-2">%</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.trades && results.trades.map((trade, i) => (
                          <tr key={i} className="border-b border-white/10 hover:bg-white/5">
                            <td className="p-2">{trade.date.split('T')[0]}</td>
                            <td className="p-2">
                              <span className={`px-2 py-1 rounded text-xs ${trade.type === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                                {trade.type}
                              </span>
                            </td>
                            <td className="text-right p-2">${trade.entry.toFixed(2)}</td>
                            <td className="text-right p-2">${trade.exit.toFixed(2)}</td>
                            <td className={`text-right p-2 font-semibold ${trade.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                            </td>
                            <td className={`text-right p-2 font-semibold ${trade.pnl_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {trade.pnl_pct > 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {results && results.error && (
              <div className="bg-red-500/20 border border-red-500 rounded-xl p-6 text-center">
                <p className="text-lg">⚠️ {results.error}</p>
                <p className="text-sm opacity-70 mt-2">Try running the backtest with different parameters</p>
              </div>
            )}
          </>
        )}

        {/* BACKTEST TAB */}
        {activeTab === 'backtest' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/10 border border-white/20 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">⚙️ Run Backtest</h3>
              <button
                onClick={runBacktest}
                disabled={loading}
                className="w-full bg-gradient-to-r from-green-500 to-emerald-500 py-4 rounded-xl font-bold text-lg hover:opacity-90 transition disabled:opacity-50"
              >
                {loading ? '⏳ Running...' : '🚀 Run Backtest'}
              </button>
              <div className="mt-4 text-sm opacity-70">
                <p>• Pair: {selectedPair}</p>
                <p>• Initial Capital: $10,000</p>
                <p>• Strategy: NYX v0.8 (HSMM + SMC)</p>
              </div>
            </div>
            
            <div className="bg-white/10 border border-white/20 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">📊 Latest Results</h3>
              {results && !results.error ? (
                <div className="space-y-3">
                  <ResultRow label="Return" value={`+${results.total_return.toFixed(2)}%`} />
                  <ResultRow label="CAGR" value={`${results.cagr.toFixed(2)}%`} />
                  <ResultRow label="Win Rate" value={`${results.win_rate.toFixed(1)}%`} />
                  <ResultRow label="Profit Factor" value={results.profit_factor.toFixed(2)} />
                  <ResultRow label="Trades" value={results.total_trades} />
                </div>
              ) : (
                <p className="opacity-70">No results yet. Run a backtest to see results.</p>
              )}
            </div>
          </div>
        )}

        {/* CHARTS TAB */}
        {activeTab === 'charts' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="📈 Price Chart">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="date" stroke="#999" tick={{fontSize: 10}} />
                  <YAxis stroke="#999" tick={{fontSize: 10}} />
                  <Tooltip contentStyle={{backgroundColor: '#1f2937', border: '1px solid #444', borderRadius: '8px'}} />
                  <Line type="monotone" dataKey="close" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="📊 Volume">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="date" stroke="#999" tick={{fontSize: 10}} />
                  <YAxis stroke="#999" tick={{fontSize: 10}} />
                  <Tooltip contentStyle={{backgroundColor: '#1f2937', border: '1px solid #444', borderRadius: '8px'}} />
                  <Bar dataKey="volume" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {results && !results.error && (
              <ChartCard title="🎯 Win/Loss Distribution">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Wins', value: results.winners },
                        { name: 'Losses', value: results.losers }
                      ]}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      dataKey="value"
                    >
                      {[0, 1].map((entry, index) => <Cell key={index} fill={COLORS[index]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
            )}
          </div>
        )}

        {/* PARAMETERS TAB */}
        {activeTab === 'parameters' && (
          <div className="bg-white/10 border border-white/20 rounded-xl p-6">
            <h3 className="text-xl font-bold mb-6">🎛️ Strategy Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ParamSlider label="SdC Threshold" value={params.sdc} onChange={(v) => setParams({...params, sdc: v})} min={2.5} max={5} step={0.1} />
              <ParamSlider label="Prob Threshold" value={params.prob} onChange={(v) => setParams({...params, prob: v})} min={0.4} max={0.7} step={0.01} />
              <ParamSlider label="Pyramid Threshold" value={params.pyramid} onChange={(v) => setParams({...params, pyramid: v})} min={0.05} max={0.2} step={0.01} />
              <ParamSlider label="Stop Loss" value={params.stop} onChange={(v) => setParams({...params, stop: v})} min={0.02} max={0.1} step={0.01} />
            </div>
            <div className="mt-6 p-4 bg-white/5 rounded-lg">
              <p className="text-sm opacity-70 mb-2">Current Parameters:</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div><span className="opacity-70">SdC:</span> <span className="font-bold">{params.sdc.toFixed(1)}</span></div>
                <div><span className="opacity-70">Prob:</span> <span className="font-bold">{params.prob.toFixed(2)}</span></div>
                <div><span className="opacity-70">Pyramid:</span> <span className="font-bold">{params.pyramid.toFixed(2)}</span></div>
                <div><span className="opacity-70">Stop:</span> <span className="font-bold">{params.stop.toFixed(2)}</span></div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const MetricCard = ({ label, value, sub }) => (
  <div className="bg-white/10 border border-white/20 rounded-xl p-4 hover:scale-105 transition">
    <div className="text-xs opacity-70 uppercase mb-1">{label}</div>
    <div className="text-2xl font-bold mb-1">{value}</div>
    {sub && <div className="text-xs opacity-70">{sub}</div>}
  </div>
);

const ChartCard = ({ title, children }) => (
  <div className="bg-white/10 border border-white/20 rounded-xl p-6">
    <h3 className="text-lg font-bold mb-4">{title}</h3>
    {children}
  </div>
);

const ParamSlider = ({ label, value, onChange, min, max, step }) => (
  <div>
    <div className="flex justify-between text-sm mb-2">
      <span>{label}</span>
      <span className="font-bold">{value.toFixed(step < 0.1 ? 2 : 1)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full"
    />
  </div>
);

const ResultRow = ({ label, value }) => (
  <div className="flex justify-between p-3 bg-white/5 rounded-lg">
    <span className="opacity-70">{label}</span>
    <span className="font-bold">{value}</span>
  </div>
);

export default App;
