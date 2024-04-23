import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "react-hot-toast";
import "react-responsive-modal/styles.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "Ratchet + Phi",
    description: "Simple demo of Phi.",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={inter.className}>
                <main className="flex flex-1 flex-col max-w-screen-xl mx-auto">
                    <Toaster />
                    <div className="flex-1">{children}</div>
                </main>
            </body>
        </html>
    );
}
