
import globals from "globals";
import pluginJs from "@eslint/js";

export default [
    {
        ignores: ["dist/"]
    },
    {
        languageOptions: {
            globals: {
                ...globals.browser,
                module: "readonly",
                require: "readonly"
            }
        }
    },
    pluginJs.configs.recommended,
    {
        rules: {
            "no-unused-vars": ["warn", { "argsIgnorePattern": "^_" }],
            "no-undef": "warn"
        }
    }
];
