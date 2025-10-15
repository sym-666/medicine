<template>
    <!-- dta页面 -->
    <!-- 药物靶向亲和力 -->
    <div class="dataContain">
        <div class="dataCard">
            <div class="dataTitle">
                <div class="dataTitleLarge">
                    Drug-Target Affinity
                </div>
                <div class="dataStep">
                    <div class="dataStepL">
                        <img class="dataStepImg" src="../../assets/images/example.png" alt="">
                        <div class="dataStepText">加载案例</div>
                    </div>
                    <div class="dataStepL2">
                        <img class="dataStepImg" src="../../assets/images/arrow.png" alt="">
                    </div>
                    <div class="dataStepL">
                        <img class="dataStepImg" src="../../assets/images/run.png" alt="">
                        <div class="dataStepText">运行</div>
                    </div>
                    <div class="dataStepL2">
                        <img class="dataStepImg" src="../../assets/images/arrow.png" alt="">
                    </div>
                    <div class="dataStepL">
                        <img class="dataStepImg" src="../../assets/images/result.png" alt="">
                        <div class="dataStepText">结果</div>
                    </div>
                </div>

                <div class="dataTitleText">
                    Drug-Target Affinity (DTA) refers to the strength of the interaction between a drug molecule and its
                    target protein, such as an enzyme, receptor, or ion channel. It is a critical parameter in drug
                    discovery and development, as it determines the efficacy and specificity of a drug. High-affinity
                    binding ensures that the drug can effectively modulate the target's activity, while low-affinity
                    interactions may lead to reduced therapeutic effects or off-target side effects.
                    DTA is typically measured using experimental techniques such as surface plasmon resonance (SPR),
                    isothermal titration calorimetry (ITC), or computational methods like molecular docking and machine
                    learning models. Understanding and optimizing DTA is essential for designing potent and selective
                    drugs, ultimately improving treatment outcomes for various diseases.
                </div>
                <div class="dataInput">Input Sequences</div>
            </div>
            <div class="dataMain">
                <div class="dataMainIput">
                    <QICard v-model:model-value="Drug" title="Drug smile:"></QICard>
                    <QICard v-model:model-value="Protein" title="Protein sequence:"></QICard>
                </div>
                <div class="dataMainOutput">
                    <QIBtn @click="dtaLoadExp" title="LoadExample"></QIBtn>
                    <QIBtn @click="dtaRun" title="RUN"></QIBtn>
                    <QIBtn @click="dtaReset" title="RESET"></QIBtn>
                </div>
                <div class="dataMainResult">
                    <div class="dataResultTitle">Result:</div>
                    <!-- <div class="antigenResult">affinity：1.5</div> -->
                    <div class="dataResult"> {{ dtaResult }}</div>
                </div>
            </div>
        </div>

    </div>




</template>

<script setup>
import { ref } from 'vue';
import QICard from '../../components/QICard.vue';
import QIBtn from '../../components/QIBtn.vue';
import { dtaUseStore } from '../../store/dta/index';
import request from '../../utils/request.js'; // 导入 request.js

const store = dtaUseStore();
const Drug = ref('');
const Protein = ref('');
const dtaResult = ref('');


const dtaLoadExp = () => {
    const dtaDrugExp = 'C1=CC(=C(C=C1CNC2=C(C(=O)C2=O)NC3=CC=NC=C3)Cl)Cl'
    const dtaProteinExp = 'MADEDLIFRLEGVDGGQSPRAGHDGDSDGDSDDEEGYFICPITDDPSSNQNVNSKVNKYYSNLTKSERYSSSGSPANSFHFKEAWKHAIQKAKHMPDPWAEFHLEDIATERATRHRYNAVTGEWLDDEVLIKMASQPFGRGAMRECFRTKKLSNFLHAQQWKGASNYVAKRYIEPVDRDVYFEDVRLQMEAKLWGEEYNRHKPPKQVDIMQMCIIELKDRPGKPLFHLEHYIEGKYIKYNSNSGFVRDDNIRLTPQAFSHFTFERSGHQLIVVDIQGVGDLYTDPQIHTETGTDFGDGNLGVRGMALFFYSHACNRICESMGLAPFDLSPRERDAVNQNTKLLQSAKTILRGTEEKCGSPQVRTLSGSRPPLLRPLSENSGDENMSDVTFDSLPSSPSSATPHSQKLDHLHWPVFSDLDNMASRDHDHLDNHRESENSGDSGYPSEKRGELDDPEPREHGHSYSNRKYESDEDSLGSSGRVCVEKWNLLNSSRLHLPRASAVALEVQRLNALDLEKKIGKSILGKVHLAMVRYHEGGRFCEKGEEWDQESAVFHLEHAANLGELEAIVGLGLMYSQLPHHILADVSLKETEENKTKGFDYLLKAAEAGDRQSMILVARAFDSGQNLSPDRCQDWLEALHWYNTALEMTDCDEGGEYDGMQDEPRYMMLAREAEMLFTGGYGLEKDPQRSGDLYTQAAEAAMEAMKGRLANQYYQKAEEAWAQMEE'
    Drug.value = dtaDrugExp;
    Protein.value = dtaProteinExp;
    store.setDtaDrug(dtaDrugExp);
    store.setDtaProtein(dtaProteinExp);
}

const dtaRun = async () => {
    try {
        const response = await request.post('/predict_dta', {
            smiles: Drug.value,
            protein: Protein.value,
        });
        console.log(response)
        dtaResult.value = response.affinity; // 假设返回的数据格式中有 result 字段
    } catch (error) {
        console.error('请求失败:', error);
    }

    console.log('dtaRun1')
    console.log(dtaResult)

}


const dtaReset = () => {
    Drug.value = '';
    Protein.value = '';
    store.setDtaDrug('');
    store.setDtaProtein('');
}



</script>


<style lang="scss" scoped>
.dataContain {
    width: 100%;
    min-height: 100vh;
    display: flex;
    justify-content: center;

    .dataCard {
        width: 80%;
        min-height: 100vh;
        background-color: #ffffff;
        margin-top: 100px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);

        .dataTitle {
            margin-top: -50px;
            width: 100%;
            min-height: 30vh;
            border-bottom: 1px solid #ccc;

            .dataStep {
                width: 100%;
                height: 150px;
                // background-color: beige;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: row;

                .dataStepL {
                    width: 100px;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    // background-color: beige;
                }

                .dataStepL2 {
                    width: 100px;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #fff;
                }

                .dataStepImg {
                    width: 60px;
                    height: 60px;
                }

                .dataStepText {
                    font-size: large;
                }
            }

            .dataTitleLarge {
                padding: 10px;
                font-size: 40px;
                margin-top: 7vh;
                margin-left: 120px;
            }

            .dataTitleText {
                width: 80%;
                background-color: #f5f7fa;
                padding: 10px;
                margin-left: 120px;
            }

            .dataInput {
                width: 100%;
                height: 50px;
                font-size: 20px;
                font-weight: 800;
                margin-left: 120px;
                margin-top: 10px;
            }
        }

        .dataMain {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;

            .dataMainIput {
                width: 100%;
                height: 40vh;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .dataMainOutput {
                width: 100%;
                height: 10vh;
                display: flex;
                justify-content: center;
                align-items: center;

            }

            .dataMainResult {
                width: 100%;
                height: 10vh;
                display: flex;
                align-items: center;
                justify-content: start;

                .dataResultTitle {
                    font-weight: 800;
                    font-size: 30px;
                    margin-left: 160px;
                }

                .dataResult {
                    font-weight: 800;
                    font-size: 30px;
                    // margin-left: 20px;

                }
            }
        }
    }

}
</style>
