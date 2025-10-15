<template>
    <!-- ADC页面 -->
    <!-- 抗体和药物亲和力 -->
    <div class="ADCContain">
        <div class="ADCCard">
            <div class="ADCTitle">
                <div class="ADCTitleLarge">
                    Antibody-Drug Conjugate
                </div>
                <div class="ADCStep">
                    <div class="ADCStepL">
                        <img class="ADCStepImg" src="../../assets/images/example.png" alt="">
                        <div class="ADCStepText">加载案例</div>
                    </div>
                    <div class="ADCStepL2">
                        <img class="ADCStepImg" src="../../assets/images/arrow.png" alt="">
                    </div>
                    <div class="ADCStepL">
                        <img class="ADCStepImg" src="../../assets/images/run.png" alt="">
                        <div class="ADCStepText">运行</div>
                    </div>
                    <div class="ADCStepL2">
                        <img class="ADCStepImg" src="../../assets/images/arrow.png" alt="">
                    </div>
                    <div class="ADCStepL">
                        <img class="ADCStepImg" src="../../assets/images/result.png" alt="">
                        <div class="ADCStepText">结果</div>
                    </div>
                </div>

                <div class="ADCTitleText">
                    Antibody-Drug Conjugates (ADCs) are a class of targeted cancer therapies that combine the
                    specificity of monoclonal antibodies with the potent cytotoxicity of chemotherapeutic drugs. ADCs
                    are designed to deliver highly toxic drugs directly to cancer cells while minimizing damage to
                    healthy tissues.
                </div>
                <div class="ADCInput">Input Sequences</div>
            </div>
            <div class="ADCMain">
                <div class="ADCMainIput">
                    <QICard v-model:model-value="adcLinker" title="linker smile"></QICard>
                    <QICard v-model:model-value="adcPlayLoad" title="playload smile"></QICard>
                    <QICard v-model:model-value="adcHeavy" title="Heavy sequence"></QICard>
                    <QICard v-model:model-value="adcLight" title="Light sequence"></QICard>
                    <QICard v-model:model-value="adcAntigen" title="Antigen sequence"></QICard>
                    <QICard v-model:model-value="adcDar" title="dar_str"></QICard>
                </div>
                <div class="ADCMainOutput">
                    <QIBtn @click="adcLoadExp" title="LoadExample"></QIBtn>
                    <QIBtn @click="adcRun" title="RUN"></QIBtn>
                    <QIBtn @click="adcReset" title="RESET"></QIBtn>
                </div>
                <div class="ADCMainResult">
                    <div class="ADCResultTitle">Result:</div>
                    <div class="ADCResult">{{ adcResult }}</div>
                </div>
            </div>
        </div>

    </div>




</template>

<script setup>
import { ref } from 'vue';
import QICard from '../../components/QICard.vue';
import QIBtn from '../../components/QIBtn.vue';
import { adcUseStore } from '../../store/adc/index';
import request from '../../utils/request.js'; // 导入 request.js

const store = adcUseStore();
const adcResult = ref('');
const adcLinker = ref('');
const adcPlayLoad = ref('');
const adcHeavy = ref('');
const adcLight = ref('');
const adcAntigen = ref('');
const adcDar = ref('')
const adcLoadExp = () => {
    const adcLinkerExp = 'O=C(O)CCCCCN1C(=O)C=CC1=O'; // 输入1：linker smile
    const adcPlayLoadExp = 'CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@@H](CC2=CC=CC=C2)C(=O)O)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC'; // 输入2：playload smile
    const adcHeavyExp = 'EVQLVESGGGLVQPGGSLRLSCAASGYTFTNFGMNWVRQAPGKGLEWVAWINTNTGEPRYAEEFKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDWDGAYFFDYWGQGTLVTVSS'; // 输入3：Heavy sequence
    const adcLightExp = 'DIQMTQSPSSLSASVGDRVTITCKASQSVSNDVAWYQQKPGKAPKLLIYFATNRYTGVPSRFSGSGYGTDFTLTISSLQPEDFATYYCQQDYSSPWTFGQGTKVEIK'; // 输入4：Light sequence
    const adcAntigenExp = 'MPGGCSRGPAAGDGRLRLARLALVLLGWVSSSSPTSSASSFSSSAPFLASAVSAQPPLPDQCPALCECSEAARTVKCVNRNLTEVPTDLPAYVRNLFLTGNQLAVLPAGAFARRPPLAELAALNLSGSRLDEVRAGAFEHLPSLRQLDLSHNPLADLSPFAFSGSNASVSAPSPLVELILNHIVPPEDERQNRSFEGMVVAALLAGRALQGLRRLELASNHFLYLPRDVLAQLPSLRHLDLSNNSLVSLTYVSFRNLTHLESLHLEDNALKVLHNGTLAELQGLPHIRVFLDNNPWVCDCHMADMVTWLKETEVVQGKDRLTCAYPEKMRNRVLLELNSADLDCDPILPPSLQTSYVFLGIVLALIGAIFLLVLYLNRKGIKKWMHNIRDACRDHMEGYHYRYEINADPRLTNLSSNSDV'; // 输入5：Antigen sequence
    const adcDarExp = '4'

    adcLinker.value = adcLinkerExp;
    adcPlayLoad.value = adcPlayLoadExp;
    adcHeavy.value = adcHeavyExp;
    adcLight.value = adcLightExp;
    adcAntigen.value = adcAntigenExp;
    adcDar.value = adcDarExp

    store.setAdcLinker(adcLinkerExp);
    store.setAdcPlayLoad(adcPlayLoadExp);
    store.setAdcHeavy(adcHeavyExp);
    store.setAdcLight(adcLightExp);
    store.setAdcAntigen(adcAntigenExp);
    store.setAdcDar(adcDarExp)
}
const adcRun = async () => {
    try {
        const response = await request.post('/predict_adc', {
            heavy_seq: adcHeavy.value,
            light_seq: adcLight.value,
            antigen_seq: adcAntigen.value,
            payload_s: adcPlayLoad.value,
            linker_s: adcLinker.value,
            dar_str: adcDar.value,
        });
        console.log(response)
        adcResult.value = response.prediction; // 假设返回的数据格式中有 result 字段
    } catch (error) {
        console.error('请求失败:', error);
    }
    console.log('adcRun1')
}

const adcReset = () => {
    adcLinker.value = '';
    adcPlayLoad.value = '';
    adcHeavy.value = '';
    adcLight.value = '';
    adcAntigen.value = '';
    adcDar.value = '';
    store.setAdcLinker('');
    store.setAdcPlayLoad('');
    store.setAdcHeavy('');
    store.setAdcLight('');
    store.setAdcDar('')
}




</script>


<style lang="scss" scoped>
.ADCContain {
    width: 100%;
    min-height: 150vh;
    display: flex;
    justify-content: center;

    .ADCCard {
        width: 80%;
        min-height: 150vh;
        background-color: #ffffff;
        margin-top: 100px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);

        .ADCTitle {
            margin-top: -50px;
            width: 100%;
            min-height: 20vh;
            border-bottom: 1px solid #ccc;

            .ADCStep {
                width: 100%;
                height: 150px;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: row;

                .ADCStepL {
                    width: 100px;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }

                .ADCStepL2 {
                    width: 100px;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #fff;
                }

                .ADCStepImg {
                    width: 60px;
                    height: 60px;
                }

                .ADCStepText {
                    font-size: large;
                }
            }

            .ADCTitleLarge {
                padding: 10px;
                font-size: 40px;
                margin-top: 7vh;
                margin-left: 120px;
            }

            .ADCTitleText {
                width: 80%;
                padding: 10px;
                margin-left: 120px;
                background-color: #f5f7f9;
            }

            .ADCInput {
                width: 100%;
                height: 50px;
                font-size: 20px;
                font-weight: 800;
                margin-left: 120px;
                margin-top: 10px;
            }
        }

        .ADCMain {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;

            .ADCMainIput {
                width: 100%;
                height: auto;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .ADCMainOutput {
                width: 100%;
                height: auto;
                display: flex;
                justify-content: center;
                align-items: center;

            }

            .ADCMainResult {
                width: 100%;
                height: auto;
                display: flex;
                align-items: center;
                justify-content: start;

                .ADCResultTitle {
                    font-weight: 800;
                    font-size: 30px;
                    margin-left: 160px;
                }

                .ADCResult {
                    font-weight: 800;
                    font-size: 30px;
                    margin-left: 20px;

                }
            }
        }
    }
}
</style>
