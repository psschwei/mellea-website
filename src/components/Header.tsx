'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { siteConfig } from '@/config/site';

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="header">
      <div className="header-inner">
        <Link href="/" className="header-logo">
          <span className="logo-bracket">[</span>
          Mellea
          <span className="logo-bracket">]</span>
        </Link>

        <nav className="header-nav">
          <Link
            href={siteConfig.docsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link"
          >
            Docs
          </Link>
          <Link
            href="/blogs"
            className={`nav-link ${pathname.startsWith('/blogs') ? 'active' : ''}`}
          >
            Blog
          </Link>
          <Link
            href={siteConfig.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link"
          >
            GitHub
          </Link>
          <Link
            href={siteConfig.docsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="nav-cta"
          >
            Get Started →
          </Link>
        </nav>
      </div>
    </header>
  );
}
