/** @type {import('tailwindcss').Config} */
module.exports = {
  //remove unused styles during build
  purge: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      fontFamily: {
        merriweather: ["Merriweather", "serif"],
      },
      backgroundImage: {
        "custom-image":
          "url('assets/images/header-bg.jpg')",
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
};

