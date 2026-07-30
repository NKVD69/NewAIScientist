export default {
  plugins: {
    // Tailwind v4 moved the PostCSS plugin into its own package.
    // The previous `tailwindcss: {}` entry is v3 syntax and is a no-op
    // against v4, so utilities were silently not being generated.
    '@tailwindcss/postcss': {},
  },
}
