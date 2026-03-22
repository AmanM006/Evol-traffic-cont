import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ATSC — Evolutionary Traffic Signal Control",
  description: "GA vs Fixed-Time side-by-side traffic simulation dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}