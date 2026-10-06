const API_BASE = `http://127.0.0.1:${import.meta.env.VITE_API_PORT || "8000"}`;

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return data;
}

export function startGame() {
  return request("/game/start", { method: "POST" });
}

export function resetGame() {
  return request("/game/reset", { method: "POST" });
}

export function playMove(row, col) {
  return request("/game/move", {
    method: "POST",
    body: JSON.stringify({ row, col }),
  });
}