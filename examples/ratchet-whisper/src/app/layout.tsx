import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import "react-responsive-modal/styles.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "Whisper by Ratchet",
    description: "Simple demo of Whisper.",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={inter.className}>
                <main className="flex flex-1 flex-col">
                    <div className="flex-1">{children}</div>
                </main>
            </body>
        </html>
    );
}
