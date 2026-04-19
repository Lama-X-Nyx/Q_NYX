import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";

const mono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NYX Cockpit",
  description: "Operator dashboard for NYX trading system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${mono.variable} h-full antialiased dark`}>
      <body className="min-h-full bg-gray-950 text-gray-100 font-mono">
        {children}
      </body>
    </html>
  );
}
