<template>
    <div class="titleBtn" @click="handleClick" @mouseenter="handleMouseEnter" @mouseleave="handleMouseLeave">
        <span class="btn-content">
            <span class="btn-text">{{ title }}</span>
            <span v-if="hasDropdown" class="arrow" :class="{ 'rotated': isOpen }">▼</span>
        </span>
        <div v-if="hasDropdown" ref="dropdownContainer">
            <slot name="dropdown"></slot>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import gsap from 'gsap';

const props = defineProps({
    title: String,
    hasDropdown: Boolean
});

const emit = defineEmits(['main-click']);

const isOpen = ref(false);
const dropdownContainer = ref(null);
let timeline;

onMounted(() => {
    if (props.hasDropdown && dropdownContainer.value) {
        const dropdownMenu = dropdownContainer.value.querySelector('.dropdown-menu');
        if (dropdownMenu) {
            gsap.set(dropdownMenu, { autoAlpha: 0, y: -10 });
            timeline = gsap.timeline({ paused: true })
                .to(dropdownMenu, {
                    autoAlpha: 1,
                    y: 0,
                    duration: 0.3,
                    ease: 'power2.out'
                });
        }
    }
});

const handleClick = () => {
    if (!props.hasDropdown) {
        emit('main-click');
    } else {
        toggleDropdown();
    }
};

const handleMouseEnter = () => {
    if (props.hasDropdown) {
        isOpen.value = true;
        timeline?.play();
    }
};

const handleMouseLeave = () => {
    if (props.hasDropdown) {
        timeline?.reverse().then(() => {
            isOpen.value = false;
        });
    }
};

const toggleDropdown = () => {
    isOpen.value = !isOpen.value;
    if (isOpen.value) {
        timeline?.play();
    } else {
        timeline?.reverse();
    }
};
</script>

<style lang="scss" scoped>
.titleBtn {
    height: 50px;
    padding: 0 25px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    position: relative;
    transition: all 0.3s ease;
    border-radius: 12px;
    margin: 0 3px;
    background: transparent;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    line-height: 1;
    vertical-align: middle;

    &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        opacity: 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    &::after {
        content: '';
        position: absolute;
        bottom: 8px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 3px;
        background: linear-gradient(45deg, #fbbf24, #f59e0b);
        border-radius: 1.5px;
        transition: width 0.3s ease;
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.4);
    }

    &:hover {
        color: #fbbf24;
        transform: translateY(-2px);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);

        &::before {
            opacity: 1;
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
        }

        &::after {
            width: 70%;
        }
    }

    &:active {
        transform: translateY(-1px);
    }

    // 确保文字始终可见
    z-index: 10;
}

.btn-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    line-height: 1;
}

.btn-text {
    display: inline-block;
    vertical-align: middle;
    line-height: 1;
}

.arrow {
    font-size: 0.8rem;
    transition: all 0.3s ease;
    color: rgba(255, 255, 255, 0.9);
    text-shadow: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
    vertical-align: middle;
}

.rotated {
    transform: rotate(180deg);
    color: #fbbf24;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
</style>
