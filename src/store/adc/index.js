import { defineStore } from "pinia";

export const adcUseStore = defineStore("main", {
  state: () => {
    return {
      adcLinker: '',
      adcPlayLoad: '',
      adcHeavy: '',
      adcLight: '',
      adcAntigen: '',
      adcDar: ''
    };
  },
  actions: {
    // 设置 Heavy Chain
    setAdcHeavy(value) {
      this.adcHeavy = value;
    },
    // 设置 Light Chain
    setAdcLight(value) {
      this.adcLight = value;
    },
    // 设置 Antigen（修正函数名称）
    setAdcAntigen(value) {
      this.adcAntigen = value;
    },
    // 设置 Linker
    setAdcLinker(value) {
      this.adcLinker = value;
    },
    // 设置 Payload
    setAdcPlayLoad(value) {
      this.adcPlayLoad = value;
    },
    // 设置 Dar
    setAdcDar(value) {
      this.adcDar = value;
    },
  },
});
