/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Light Mode Defaults
                app: {
                    bg: '#FFFFFF',
                    text: '#111111',
                    subtext: '#757575',
                    divider: '#EEEEEE',
                },
                accent: {
                    DEFAULT: '#2962FF',
                    hover: '#0039Cb', // Darker shade for hover
                },
                // Dark Mode Overrides (used via utilities usually, or CSS vars. 
                // But for Tailwind config to support both easily, it's often easier to just use standard classes like dark:bg-zinc-900.
                // However, user gave SPECIFIC hex codes.
                // I will define them as specific palette colors and use them in classes.

                // Dark Mode specific palette
                dark: {
                    bg: '#121212',
                    text: '#E0E0E0',
                    subtext: '#9E9E9E',
                    // Accent is same
                }
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
