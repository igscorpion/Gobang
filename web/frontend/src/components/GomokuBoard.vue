<template>
  <div class="board-wrap">
    <div
      class="board"
      :style="{ gridTemplateColumns: `repeat(${size}, var(--cell-size))` }"
    >
      <button
        v-for="cell in cells"
        :key="cell.key"
        class="cell"
        :class="cell.className"
        :disabled="disabled || cell.stone !== 0"
        :aria-label="cell.label"
        @click="$emit('place', cell.row, cell.col)"
      >
        <span v-if="cell.stone !== 0" class="stone" :class="cell.stoneClass" />
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: "GomokuBoard",
  props: {
    board: {
      type: Array,
      required: true,
    },
    lastMove: {
      type: Object,
      default: null,
    },
    disabled: {
      type: Boolean,
      default: false,
    },
  },
  emits: ["place"],
  computed: {
    size() {
      return this.board.length || 9;
    },
    cells() {
      const last = this.lastMove;
      const items = [];
      for (let row = 0; row < this.size; row += 1) {
        for (let col = 0; col < this.size; col += 1) {
          const stone = this.board[row][col];
          const isLast =
            last && last.row === row && last.col === col;
          items.push({
            key: `${row}-${col}`,
            row,
            col,
            stone,
            label: `第 ${row + 1} 行第 ${col + 1} 列`,
            className: {
              last: isLast,
              "row-start": col === 0,
              "col-start": row === 0,
            },
            stoneClass: stone === 1 ? "black" : "white",
          });
        }
      }
      return items;
    },
  },
};
</script>
