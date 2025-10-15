import { defineStore } from "pinia";

export const dtaUseStore = defineStore("main", {
  state: () => {
    return {
      dtaDrug: '',
      dtaProtein: '',
    };
  },
  actions: {
    setDtaDrug(value) {
      this.dtaDrug = value;
    },
    setDtaProtein(value) {
      this.dtaProtein = value;
    },
  },
});