import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Physical AI Playground",
  description: "Control a simulated robot with natural language and watch its agent loop unfold.",
  openGraph: {
    title: "Physical AI Playground",
    description: "Observe, decide, and act with a browser-based robotics agent.",
    images: [{ url: "/og.png", width: 1733, height: 909 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Physical AI Playground",
    description: "Observe, decide, and act with a browser-based robotics agent.",
    images: ["/og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
