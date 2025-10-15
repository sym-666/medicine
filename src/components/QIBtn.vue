<template>
    <div class="QIBtn">
        {{title}}
    </div>
</template>

<script setup>
import { defineProps } from 'vue';
defineProps({
    title: {
        type: String,
        required: true
    }
});
</script>

<style lang="scss" scoped>
 .QIBtn {
    width: 15%;
    height: 8vh;
    border-radius: 10px;
    background-color: #5cc1ef;
    color: white;
    font-size: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-left: 25px;
    cursor: pointer;
 }


</style>
