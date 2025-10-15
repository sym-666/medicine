<template>
    <!-- 抗原抗体亲和力检测页面 -->
    <div class="antigenContain">
        <div class="antigenCard">
            <div class="antigenTitle">
                <div class="antigenTitleLarge">
                    Antigen-Antibody Affinity Detection
                </div>
                <div class="antigenStep">
                    <div class="antigenStepL">
                        <img class="antigenStepImg" src="../../assets/images/example.png" alt="">
                        <div class="antigenStepText">加载案例</div>
                    </div>
                    <div class="antigenStepL2">
                        <img class="antigenStepImg" src="../../assets/images/arrow.png" alt="">
                    </div>
                    <div class="antigenStepL">
                        <img class="antigenStepImg" src="../../assets/images/run.png" alt="">
                        <div class="antigenStepText">运行</div>
                    </div>
                    <div class="antigenStepL2">
                        <img class="antigenStepImg" src="../../assets/images/arrow.png" alt="">
                    </div>
                    <div class="antigenStepL">
                        <img class="antigenStepImg" src="../../assets/images/result.png" alt="">
                        <div class="antigenStepText">结果</div>
                    </div>
                </div>
                <div class="antigenTitleText">
                    Antigen-antibody affinity detection is a critical technique in immunology and biotechnology, used to
                    measure the strength of the interaction between an antigen and its corresponding antibody. This
                    process is essential for understanding immune responses, developing diagnostic assays, and
                    engineering therapeutic antibodies. High-affinity interactions indicate a strong and specific
                    binding, which is crucial for the efficacy of antibodies in targeting pathogens or diseased cells.
                    Various methods, such as surface plasmon resonance (SPR), enzyme-linked immunosorbent assay (ELISA),
                    and fluorescence-based assays, are employed to quantify this affinity, providing valuable insights
                    into the molecular dynamics of immune recognition.
                </div>
                <div class="antigenInput">Input Sequences</div>
            </div>
            <div class="antigenMain">
                <div class="antigenMainIput">
                    <QICard v-model:model-value="Heavy" title="Heavy Chain Sequence"></QICard>
                    <QICard v-model:model-value="Light" title="Light Chain Sequence"></QICard>
                    <QICard v-model:model-value="Anti" title="Antigen Sequence"></QICard>
                </div>
                <div class="antigenMainOutput">
                    <QIBtn @click="antiLoadExp" title="LoadExample"></QIBtn>
                    <QIBtn title="RUN" @click="antiRun"></QIBtn>
                    <QIBtn title="RESET" @click="antiReset"></QIBtn>
                </div>
                <div class="antigenMainResult">
                    <div class="antigenResultTitle">Result:</div>
                    <div class="antigenResult">{{ antiResult }}</div>
                </div>
            </div>
        </div>
    </div>




</template>

<script setup>
import QICard from '../../components/QICard.vue';
import QIBtn from '../../components/QIBtn.vue';
import { ref, watch } from 'vue';
import { antiUseStore } from '../../store/anti/index.js'
import request from '../../utils/request.js'; // 导入 request.js

const store = antiUseStore();
const Heavy = ref('');
const Light = ref('');
const Anti = ref('');
const antiResult = ref('');

const antiLoadExp = () => {
    const heavyExp = 'QVQLQESGPGLVKPSQTLSLTCSFSGFSLSTSGMGVGWIRQPSGKGLEWLAHIWWDGDESYNPSLKSRLTISKDTSKNQVSLKITSVTAADTAVYFCARNRYDPPWFVDWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEP'
    const lightExp = 'PVRSLNCTLRDSQQKSLVMSGPYELKALHLQGQDMEQQVVFSMSFVQGEESNDKIPVALGLKEKNLYLSCVLKDDKPTLQLESVDPKNYPKKKMEKRFVFNKIEINNKLEFESAQFPNWYISTSQAENMPVFLGGTKGGQDITDFTMQFV'
    const antiExp = 'DIQMTQSTSSLSASVGDRVTITCRASQDISNYLSWYQQKPGKAVKLLIYYTSKLHSGVPSRFSGSGSGTDYTLTISSLQQEDFATYFCLQGKMLPWTFGQGTKLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGE'
    Heavy.value = heavyExp;
    Light.value = lightExp;
    Anti.value = antiExp;
    store.setHeavy(heavyExp);
    store.setLight(lightExp);
    store.setAnti(antiExp);
}

const antiRun = async () => {
    try {
        const response = await request.post('/predict', {
            seq_light: Light.value,
            seq_heavy: Heavy.value,
            seq_antigen: Anti.value,
        });
        console.log(response)
        antiResult.value = response.prediction; // 假设返回的数据格式中有 result 字段
    } catch (error) {
        console.error('请求失败:', error);
    }

    // console.log('antiRun1')
    // console.log(antiResult)
}


const antiReset = () => {
    Heavy.value = '';
    Light.value = '';
    Anti.value = '';
    store.setHeavy('');
    store.setLight('');
    store.setAnti('');
}


watch(Heavy, (newValue) => {
    store.setHeavy(newValue);
})
watch(Light, (newValue) => {
    store.setLight(newValue);
})
watch(Anti, (newValue) => {
    store.setAnti(newValue);
})



</script>


<style lang="scss" scoped>
.antigenContain {
    width: 100%;
    min-height: 120vh;
    display: flex;
    justify-content: center;

    .antigenCard {
        margin-top: 100px;
        width: 80%;
        min-height: 120vh;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);

        .antigenTitle {
            margin-top: -50px;
            width: 100%;
            min-height: 25vh;
            border-bottom: 1px solid #ccc;

            .antigenStep {
                width: 100%;
                height: 150px;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: row;

                .antigenStepL {
                    width: 100px;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }

                .antigenStepL2 {
                    width: 100px;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #fff;
                }

                .antigenStepImg {
                    width: 60px;
                    height: 60px;
                }

                .antigenStepText {
                    font-size: large;
                }
            }

            .antigenTitleLarge {
                padding: 10px;
                font-size: 40px;
                margin-top: 6vh;
                margin-left: 120px;
            }

            .antigenTitleText {
                width: 80%;
                padding: 10px;
                margin-left: 120px;
                background-color: #f5f7f9;
            }

            .antigenInput {
                width: 100%;
                height: 50px;
                font-size: 20px;
                font-weight: 800;
                margin-left: 120px;
                margin-top: 10px;
            }
        }

        .antigenMain {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;

            .antigenMainIput {
                width: 100%;
                height: 40vh;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .antigenMainOutput {
                width: 100%;
                height: 10vh;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 10vh;
            }

            .antigenMainResult {
                width: 100%;
                height: 10vh;
                display: flex;
                align-items: center;
                justify-content: start;

                .antigenResultTitle {
                    font-weight: 800;
                    font-size: 30px;
                    margin-left: 120px;
                }

                .antigenResult {
                    font-weight: 800;
                    font-size: 30px;
                    margin-left: 20px;
                }
            }
        }
    }
}
</style>
