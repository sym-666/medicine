<template>
    <div class="titleBtn" @click="handleClick" @mouseenter="handleMouseEnter" @mouseleave="handleMouseLeave">
        {{ title }}
        <span v-if="hasDropdown" class="arrow" :class="{ 'rotated': isOpen }">▼</span>
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
    height: 100%;
    padding: 0 25px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #333;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    position: relative;
    transition: color 0.3s ease;

    &::after {
        content: '';
        position: absolute;
        bottom: 15px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 2px;
        background-color: #1e3a8a;
        transition: width 0.3s ease;
    }
}

.titleBtn:hover {
    color: #1e3a8a;

    &::after {
        width: 80%;
    }
}

.arrow {
    margin-left: 8px;
    font-size: 0.7rem;
    transition: transform 0.3s ease;
}

.rotated {
    transform: rotate(180deg);
}
</style>
