import { defineStore } from "pinia"

export const antiUseStore = defineStore("main", {
  state: () => {
    return {
      Heavy: '',  
      Light: '',
      Anti:'',
    }
  },
  actions: {
    setHeavy(value) {
      this.Heavy = value
    },
    setLight(value) {
      this.Light = value
    },
    setAnti(value) {
      this.Anti = value
    },
  }
})