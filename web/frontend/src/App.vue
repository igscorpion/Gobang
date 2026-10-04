<template>
  <div class="page">
    <header class="header">
      <h1>RL Gomoku</h1>
      <p class="subtitle">人机对战 Demo · 当前 AI 为 RandomAgent</p>
    </header>

    <p class="status" :class="{ error: Boolean(error), over: gameOver }">
      {{ statusText }}
    </p>

    <GomokuBoard
      v-if="board.length"
      :board="board"
      :last-move="lastMove"
      :disabled="busy || gameOver"
      @place="onPlace"
    />

    <div class="actions">
      <button class="btn" :disabled="busy" @click="onReset">重新开始</button>
    </div>
  </div>
</template>

<script>
import GomokuBoard from "./components/GomokuBoard.vue";
import { playMove, resetGame, startGame } from "./api/game.js";

export default {
  name: "App",
  components: { GomokuBoard },
  data() {
    return {
      board: [],
      lastMove: null,
      winner: null,
      gameOver: false,
      busy: false,
      error: "",
    };
  },
  computed: {
    statusText() {
      if (this.error) {
        return this.error;
      }
      if (this.winner === "black") {
        return "黑棋（你）获胜";
      }
      if (this.winner === "white") {
        return "白棋（AI）获胜";
      }
      if (this.winner === "draw") {
        return "和棋";
      }
      if (this.busy) {
        return "AI 落子中…";
      }
      return "轮到你（黑棋）";
    },
  },
  async mounted() {
    await this.bootstrap();
  },
  methods: {
    applyState(data) {
      this.board = data.board;
      this.lastMove = data.last_move;
      this.winner = data.winner;
      this.gameOver = data.game_over;
    },
    async bootstrap() {
      this.busy = true;
      this.error = "";
      try {
        this.applyState(await startGame());
      } catch (err) {
        this.error = `无法连接后端：${err.message}。请确认 FastAPI 已在 http://127.0.0.1:8000 运行。`;
      } finally {
        this.busy = false;
      }
    },
    async onPlace(row, col) {
      if (this.busy || this.gameOver) {
        return;
      }
      if (this.board[row][col] !== 0) {
        return;
      }
      this.busy = true;
      this.error = "";
      try {
        this.applyState(await playMove(row, col));
      } catch (err) {
        this.error = err.message;
      } finally {
        this.busy = false;
      }
    },
    async onReset() {
      this.busy = true;
      this.error = "";
      try {
        this.applyState(await resetGame());
      } catch (err) {
        this.error = err.message;
      } finally {
        this.busy = false;
      }
    },
  },
};
</script>
