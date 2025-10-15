import { defineStore } from 'pinia';

export const antiUseStore = defineStore('anti', {
    state: () => ({
        Heavy: '',
        Light: '',
        Anti: ''
    }),
    actions: {
        setHeavy(value) {
            this.Heavy = value;
        },
        setLight(value) {
            this.Light = value;
        },
        setAnti(value) {
            this.Anti = value;
        }
    }
});
