import { spawn } from "node:child_process";

const command = process.argv[2];
if (!command) {
  throw new Error("Expected a vinext command (dev, build, or start).");
}

const child = spawn(
  process.execPath,
  ["node_modules/vinext/dist/cli.js", command],
  {
    stdio: "inherit",
    env: {
      ...process.env,
      WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
      ...(command === "dev" && process.platform === "win32"
        ? { SITES_USE_CLOUDFLARE_RUNTIME: "0" }
        : {}),
    },
  },
);

child.on("exit", (code) => process.exit(code ?? 1));
child.on("error", (error) => {
  console.error(error);
  process.exit(1);
});
