import type { Metadata } from 'next';
import './globals.css';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { siteConfig } from '@/config/site';

export const metadata: Metadata = {
  metadataBase: new URL(siteConfig.url),
  title: {
    default: 'Mellea — Reliable, Testable LLM Output for Python',
    template: '%s | Mellea',
  },
  description: siteConfig.description,
  openGraph: {
    type: 'website',
    siteName: siteConfig.name,
    title: 'Mellea — Reliable, Testable LLM Output for Python',
    description: siteConfig.description,
    url: siteConfig.url,
    images: [{ url: '/images/og-hero.png', width: 2064, height: 1104 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Mellea — Reliable, Testable LLM Output for Python',
    description: siteConfig.description,
    images: ['/images/og-hero.png'],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Header />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
