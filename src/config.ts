import type {
  CommentConfig,
  LicenseConfig,
  LinkConfig,
  NavBarConfig,
  ProfileConfig,
  SiteConfig,
} from './types/config'
import { LinkPreset } from './types/config'

export const siteConfig: SiteConfig = {
  title: "rooger's blog",
  subtitle: 'Always Look Forward',
  lang: 'zh_CN', // 'en', 'zh_CN', 'zh_TW', 'ja'
  themeColor: {
    hue: 250, // Default hue for the theme color, from 0 to 360. e.g. red: 0, teal: 200, cyan: 250, pink: 345
    fixed: false, // Hide the theme color picker for visitors
  },
  banner: {
    enable: true,
    src: 'assets/images/blog-banner.jpg', // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  },
  favicon: [
    // Leave this array empty to use the default favicon
    // {
    //   src: '/favicon/icon.png',    // Path of the favicon, relative to the /public directory
    //   theme: 'light',              // (Optional) Either 'light' or 'dark', set only if you have different favicons for light and dark mode
    //   sizes: '32x32',              // (Optional) Size of the favicon, set only if you have favicons of different sizes
    // }
  ],
}

export const navBarConfig: NavBarConfig = {
  links: [
    LinkPreset.Home,
    LinkPreset.Archive,
    LinkPreset.About,
    {
      name: 'GitHub',
      url: 'https://github.com/rogerchen2005/', // Internal links should not include the base path, as it is automatically added
      external: true, // Show an external link icon and will open in a new tab
    },
    {
      name: 'Site',
      url: 'https://app.cast1e.top', // Internal links should not include the base path, as it is automatically added
      external: true, // Show an external link icon and will open in a new tab
    },
  ],
}

export const profileConfig: ProfileConfig = {
  avatar: 'assets/images/blog-avatar.jpg', // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  name: 'cast1e',
  bio: 'If you shed tears when you miss the sun, you also miss the stars. --Tagore',
  links: [
    {
      name: 'Twitter',
      icon: 'fa6-brands:twitter', // Visit https://icones.js.org/ for icon codes
      // You will need to install the corresponding icon set if it's not already included
      // `pnpm add @iconify-json/<icon-set-name>`
      url: 'https://twitter.com/cast1e_pTr',
    },
    {
      name: 'Steam',
      icon: 'fa6-brands:steam',
      url: 'https://steamcommunity.com/profiles/76561198807399847/',
    },
    {
      name: 'GitHub',
      icon: 'fa6-brands:github',
      url: 'https://github.com/rogerchen2005/',
    },
    {
      name: 'QQ',
      icon: 'fa6-brands:qq',
      url: 'https://qm.qq.com/cgi-bin/qm/qr?k=ilymCBCJbH2U4zrHWiFWOjCDv5Zjp_KV',
    },
  ],
}

export const licenseConfig: LicenseConfig = {
  enable: true,
  name: 'CC BY-NC-SA 4.0',
  url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
}

export const linkConfig: LinkConfig = {
  links: [
    {
      name: '末荼の大学生活',
      url: 'https://chatter-barber-020.notion.site/40075ca669a9460d9c915b1e564c83c2?v=ea79d5524f6546eba94e3da5efb93f1e',
    },
    {
      name: '芙芙的小蛋糕',
      url: 'https://wht222.github.io/myblog',
    },
  ],
}

export const commentConfig: CommentConfig = {
  serverURL: 'https://comment.cast1e.top',
  emoji: [
    '//unpkg.com/@waline/emojis@1.4.0/tieba',
    '//unpkg.com/@waline/emojis@1.4.0/bmoji',
  ],
}
