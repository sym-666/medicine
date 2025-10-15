<template>
    <div class="QIcard">
        <div class="QITitle">
            {{ title }}
        </div>
        <div class="QIInput">
            <textarea :value="modelValue" class="QIInputText" placeholder="请输入" @input="updateValue($event.target.value)"></textarea>
        </div>
    </div>
</template>

<script setup>
import { defineProps,defineEmits } from 'vue';
defineProps({
    title: {
        type: String,
        required: true
    },
    modelValue: {
        type: String,
        required: true
    },
    
});


const emit = defineEmits(['update:modelValue']);
const updateValue = (value) => {
    emit('update:modelValue', value);
}


</script>

<style lang="scss" scoped>
.QIcard {
    width: 80%;
    min-height: 17vh;
    background-color: #ffffff;
   
    .QITitle {
        padding: 10px;
    }
    .QIInput {
        width: 100%;
        display: flex;
        justify-content: center;
        .QIInputText {
            width: 99%;
            height: 10vh;
            border: 1px solid #dddddd;
            border-radius: 5px;
        }
    }
}

</style>
