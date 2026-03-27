import { describe, it, expect } from 'vitest';
import { getAllBlogs, getBlog, getAllBlogSlugs } from '@/lib/blogs';

describe('getAllBlogs', () => {
  it('returns a non-empty array', () => {
    const blogs = getAllBlogs();
    expect(Array.isArray(blogs)).toBe(true);
    expect(blogs.length).toBeGreaterThan(0);
  });

  it('returns blogs sorted by date descending', () => {
    const blogs = getAllBlogs();
    for (let i = 0; i < blogs.length - 1; i++) {
      expect(blogs[i].date >= blogs[i + 1].date).toBe(true);
    }
  });

  it('each blog has required fields', () => {
    for (const blog of getAllBlogs()) {
      expect(blog.slug).toBeTruthy();
      expect(blog.title).toBeTruthy();
      expect(blog.date).toBeTruthy();
      expect(blog.author).toBeTruthy();
      expect(Array.isArray(blog.tags)).toBe(true);
    }
  });
});

describe('getBlog', () => {
  it('returns content for a valid slug', () => {
    const blog = getBlog('thinking-about-ai');
    expect(blog).not.toBeNull();
    expect(blog?.title).toBeTruthy();
    expect(blog?.content).toBeTruthy();
  });

  it('returns null for an unknown slug', () => {
    expect(getBlog('does-not-exist')).toBeNull();
  });
});

describe('getAllBlogSlugs', () => {
  it('matches the slugs from getAllBlogs', () => {
    const slugs = getAllBlogSlugs();
    const blogs = getAllBlogs();
    expect(slugs.length).toBe(blogs.length);
    for (const blog of blogs) {
      expect(slugs).toContain(blog.slug);
    }
  });
});
