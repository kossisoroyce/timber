import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Timber',
  tagline: 'Ollama for classical ML models. Compile and serve tree-based models at native speed.',
  favicon: 'img/favicon.ico',

  url: 'https://kossisoroyce.github.io',
  baseUrl: '/timber/',

  organizationName: 'kossisoroyce',
  projectName: 'timber',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/kossisoroyce/timber/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/timber-social-card.png',
    navbar: {
      title: 'Timber',
      logo: {
        alt: 'Timber Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://pypi.org/project/timber-compiler/',
          label: 'PyPI',
          position: 'right',
        },
        {
          href: 'https://github.com/kossisoroyce/timber',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/docs/getting-started'},
            {label: 'API Reference', to: '/docs/api-reference/cli'},
            {label: 'Examples', to: '/docs/examples/xgboost'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub Issues', href: 'https://github.com/kossisoroyce/timber/issues'},
            {label: 'Contributing', href: 'https://github.com/kossisoroyce/timber/blob/main/CONTRIBUTING.md'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'PyPI', href: 'https://pypi.org/project/timber-compiler/'},
            {label: 'GitHub', href: 'https://github.com/kossisoroyce/timber'},
            {label: 'Paper', href: 'https://github.com/kossisoroyce/timber/blob/main/paper/timber_paper.pdf'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Kossiso Royce / Electricsheep Africa. Apache-2.0 License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'c', 'python', 'json', 'toml'],
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
