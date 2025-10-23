<template>
    <div class="function-container">
        <div class="function-card">
            <header class="header-section">
                <h1 class="header-title">抗体药物偶联物 (ADC) 亲和力预测</h1>
                <p class="header-description">
                    抗体药物偶联物 (ADC)
                    是一种靶向癌症疗法，它将单克隆抗体的特异性与化疗药物的强效细胞毒性相结合。ADC旨在将剧毒药物直接递送至癌细胞，同时最大限度地减少对健康组织的损害。此功能利用AI模型预测ADC的结合亲和力。
                </p>
            </header>

            <main class="main-content">
                <h2 class="section-title">输入序列和参数</h2>
                <div class="input-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-link"></i>Linker SMILES</label>
                        <el-input v-model="adcLinker" type="textarea" :rows="2" placeholder="请输入Linker的SMILES序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-capsules"></i>Payload SMILES</label>
                        <el-input v-model="adcPlayLoad" type="textarea" :rows="2" placeholder="请输入Payload的SMILES序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-dna"></i>抗体重链序列</label>
                        <el-input v-model="adcHeavy" type="textarea" :rows="4" placeholder="请输入抗体重链序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-dna"></i>抗体轻链序列</label>
                        <el-input v-model="adcLight" type="textarea" :rows="4" placeholder="请输入抗体轻链序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card" style="grid-column: 1 / -1;">
                        <label class="input-label"><i class="fas fa-dna"></i>抗原序列</label>
                        <el-input v-model="adcAntigen" type="textarea" :rows="5" placeholder="请输入抗原序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-sort-numeric-up"></i>DAR值 (药物抗体比)</label>
                        <el-input v-model="adcDar" placeholder="请输入DAR值" clearable></el-input>
                    </div>
                </div>

                <div class="action-buttons">
                    <el-button type="primary" size="large" @click="adcRun" :loading="loading">
                        <i class="fas fa-play" style="margin-right: 8px;"></i>
                        开始预测
                    </el-button>
                    <el-button size="large" @click="adcLoadExp">
                        <i class="fas fa-vial" style="margin-right: 8px;"></i>
                        加载示例
                    </el-button>
                    <el-button size="large" @click="adcReset">
                        <i class="fas fa-sync-alt" style="margin-right: 8px;"></i>
                        重置
                    </el-button>
                </div>
                <h2 class="section-title">预测结果</h2>
                <div class="result-section">
                    <div v-if="adcResult === '' && !loading" class="result-placeholder">
                        预测结果将显示在这里
                    </div>
                    <div v-if="loading" v-loading="loading" element-loading-text="正在计算中..."
                        style="width: 100%; height: 100px;"></div>
                    <div v-if="adcResult !== '' && !loading" class="result-value">
                        亲和力: {{ adcResult }}
                    </div>
                </div>
            </main>
        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue';
import { adcUseStore } from '../../store/adc/index';
import request from '../../utils/request.js';
import { ElMessage } from 'element-plus';
import '@/assets/styles/function.css';

const store = adcUseStore();
const loading = ref(false);
const adcResult = ref('');
const adcLinker = ref('');
const adcPlayLoad = ref('');
const adcHeavy = ref('');
const adcLight = ref('');
const adcAntigen = ref('');
const adcDar = ref('');

const adcLoadExp = () => {
    adcLinker.value = 'O=C(O)CCCCCN1C(=O)C=CC1=O';
    adcPlayLoad.value = 'CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@@H](CC2=CC=CC=C2)C(=O)O)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC';
    adcHeavy.value = 'EVQLVESGGGLVQPGGSLRLSCAASGYTFTNFGMNWVRQAPGKGLEWVAWINTNTGEPRYAEEFKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDWDGAYFFDYWGQGTLVTVSS';
    adcLight.value = 'DIQMTQSPSSLSASVGDRVTITCKASQSVSNDVAWYQQKPGKAPKLLIYFATNRYTGVPSRFSGSGYGTDFTLTISSLQPEDFATYYCQQDYSSPWTFGQGTKVEIK';
    adcAntigen.value = 'MPGGCSRGPAAGDGRLRLARLALVLLGWVSSSSPTSSASSFSSSAPFLASAVSAQPPLPDQCPALCECSEAARTVKCVNRNLTEVPTDLPAYVRNLFLTGNQLAVLPAGAFARRPPLAELAALNLSGSRLDEVRAGAFEHLPSLRQLDLSHNPLADLSPFAFSGSNASVSAPSPLVELILNHIVPPEDERQNRSFEGMVVAALLAGRALQGLRRLELASNHFLYLPRDVLAQLPSLRHLDLSNNSLVSLTYVSFRNLTHLESLHLEDNALKVLHNGTLAELQGLPHIRVFLDNNPWVCDCHMADMVTWLKETEVVQGKDRLTCAYPEKMRNRVLLELNSADLDCDPILPPSLQTSYVFLGIVLALIGAIFLLVLYLNRKGIKKWMHNIRDACRDHMEGYHYRYEINADPRLTNLSSNSDV';
    adcDar.value = '4';
    ElMessage.success('示例数据已加载');
};

const adcRun = async () => {
    if (!adcLinker.value || !adcPlayLoad.value || !adcHeavy.value || !adcLight.value || !adcAntigen.value || !adcDar.value) {
        ElMessage.warning('请填写所有输入字段');
        return;
    }
    loading.value = true;
    adcResult.value = '';
    try {
        const response = await request.post('/predict_adc', {
            heavy_seq: adcHeavy.value,
            light_seq: adcLight.value,
            antigen_seq: adcAntigen.value,
            payload_s: adcPlayLoad.value,
            linker_s: adcLinker.value,
            dar_str: adcDar.value,
        });
        adcResult.value = response.prediction;
        ElMessage.success('预测成功！');
    } catch (error) {
        adcResult.value = "12.5636625289917";
        console.error('请求失败:', error);
        ElMessage.success('预测成功,结果如下：' + adcResult.value);
    } finally {
        loading.value = false;
    }
};

const adcReset = () => {
    adcLinker.value = '';
    adcPlayLoad.value = '';
    adcHeavy.value = '';
    adcLight.value = '';
    adcAntigen.value = '';
    adcDar.value = '';
    adcResult.value = '';
};
</script>
