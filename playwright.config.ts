import { defineConfig, devices } from '@playwright/test';

const port = process.env.CI ? 3000 : 4000;

export default defineConfig({
  testDir: './tests/e2e',
  outputDir: './playwright-results',
  reporter: [['html', { outputFolder: 'playwright-report', open: 'never' }]],
  use: {
    baseURL: `http://localhost:${port}`,
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: process.env.CI ? 'npx serve ./out -l 3000' : 'npm run dev',
    port,
    reuseExistingServer: !process.env.CI,
  },
});
