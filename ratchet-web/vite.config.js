import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import config from "./webdriver.json";

const wasmContentTypePlugin = {
  name: "wasm-content-type-plugin",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url.endsWith(".wasm")) {
        res.setHeader("Content-Type", "application/wasm");
      }
      next();
    });
  },
};

const plugins = [wasm(), topLevelAwait(), wasmContentTypePlugin];
export default defineConfig({
  test: {
    browser: {
      enabled: true,
      headless: true,
      name: "chrome",
      providerOptions: {
        capabilities: config, 
      }
    },
    testTimeout: 30000,
  },
  server: {
    fs: {
      // Allow serving files from target directory
      allow: ["../../target/pkg", "."],
    },
  },
  plugins: plugins,
  worker: {
    plugins: plugins,
  },
});
