<template>
    <div class="titleBtn" @click="toggleDropdown">
        {{ title }}
        <span v-if="props.hasDropdown" class="arrow" :class="{ rotated: isOpen }">▼</span>
    </div>

    <div v-if="props.hasDropdown && isOpen" class="dropdown-menu" @click.stop="isOpen = false">
        <slot name="dropdown"></slot>
    </div>
</template>

<script setup>
import { ref } from 'vue';

// 通过 defineProps 获取 props
const props = defineProps({
    title: String,
    hasDropdown: Boolean
});

const isOpen = ref(false);
const emit = defineEmits(['click']);

const toggleDropdown = () => {
    if (!props.hasDropdown) {
        emit('click');
    }
    isOpen.value = !isOpen.value;
};
</script>

<style lang="scss" scoped>
.titleBtn {
    height: 100%;
    width: 25%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: black;
    cursor: pointer;
    position: relative;
    transition: all 0.3s ease;
}

.titleBtn:hover {
    color: rebeccapurple;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
}

.arrow {
    margin-left: 5px;
    transition: transform 0.3s ease;
}

.rotated {
    transform: rotate(180deg);
}

/* 下拉菜单 */
.dropdown-menu {
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: white;
    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    width: 120px;
    // padding: 5px 0;
    z-index: 1000;
}

.menu-item {
    padding: 8px 12px;
    text-align: center;
    cursor: pointer;
    transition: background 0.3s ease;
}

.menu-item:hover {
    background: #eee;
}
</style>
