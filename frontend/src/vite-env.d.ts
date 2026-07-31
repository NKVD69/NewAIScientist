/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin. Defaults to http://localhost:8000. */
  readonly VITE_API_URL?: string;
  /** Sent as X-API-Key when the backend requires authentication. */
  readonly VITE_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
