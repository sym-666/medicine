<template>
    <div class="function-container">
        <div class="function-card">
            <header class="header-section">
                <h1 class="header-title">药物靶点亲和力 (DTA) 预测</h1>
                <p class="header-description">
                    药物靶点亲和力 (DTA)
                    是指药物分子与其靶蛋白（如酶、受体或离子通道）之间相互作用的强度。它是药物发现和开发中的一个关键参数，因为它决定了药物的功效和特异性。利用先进的AI模型，我们可以根据药物的SMILES序列和靶点的蛋白质序列，快速预测它们的结合亲和力。
                </p>
            </header>

            <main class="main-content">
                <h2 class="section-title">输入序列</h2>
                <div class="input-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-pills"></i>药物 SMILES</label>
                        <el-input v-model="drug" type="textarea" :rows="5" placeholder="请输入药物的SMILES序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-dna"></i>靶点蛋白质序列</label>
                        <el-input v-model="protein" type="textarea" :rows="5" placeholder="请输入靶点的蛋白质序列"
                            clearable></el-input>
                    </div>
                </div>

                <div class="action-buttons">
                    <el-button type="primary" size="large" @click="dtaRun" :loading="loading">
                        <i class="fas fa-play" style="margin-right: 8px;"></i>
                        开始预测
                    </el-button>
                    <el-button size="large" @click="dtaLoadExp">
                        <i class="fas fa-vial" style="margin-right: 8px;"></i>
                        加载示例
                    </el-button>
                    <el-button size="large" @click="dtaReset">
                        <i class="fas fa-sync-alt" style="margin-right: 8px;"></i>
                        重置
                    </el-button>
                </div>

                <h2 class="section-title">预测结果</h2>
                <div class="result-section">
                    <div v-if="dtaResult === '' && !loading" class="result-placeholder">
                        预测结果将显示在这里
                    </div>
                    <div v-if="loading" v-loading="loading" element-loading-text="正在计算中..."
                        style="width: 100%; height: 100px;"></div>
                    <div v-if="dtaResult !== '' && !loading" class="result-value">
                        亲和力: {{ dtaResult }}
                    </div>
                </div>
            </main>
        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue';
import { dtaUseStore } from '../../store/dta/index';
import request from '../../utils/request.js';
import { ElMessage } from 'element-plus';
import '@/assets/styles/function.css';

const store = dtaUseStore();
const drug = ref('');
const protein = ref('');
const dtaResult = ref('');
const loading = ref(false);

const dtaLoadExp = () => {
    drug.value = 'C1=CC(=C(C=C1CNC2=C(C(=O)C2=O)NC3=CC=NC=C3)Cl)Cl';
    protein.value = 'MADEDLIFRLEGVDGGQSPRAGHDGDSDGDSDDEEGYFICPITDDPSSNQNVNSKVNKYYSNLTKSERYSSSGSPANSFHFKEAWKHAIQKAKHMPDPWAEFHLEDIATERATRHRYNAVTGEWLDDEVLIKMASQPFGRGAMRECFRTKKLSNFLHAQQWKGASNYVAKRYIEPVDRDVYFEDVRLQMEAKLWGEEYNRHKPPKQVDIMQMCIIELKDRPGKPLFHLEHYIEGKYIKYNSNSGFVRDDNIRLTPQAFSHFTFERSGHQLIVVDIQGVGDLYTDPQIHTETGTDFGDGNLGVRGMALFFYSHACNRICESMGLAPFDLSPRERDAVNQNTKLLQSAKTILRGTEEKCGSPQVRTLSGSRPPLLRPLSENSGDENMSDVTFDSLPSSPSSATPHSQKLDHLHWPVFSDLDNMASRDHDHLDNHRESENSGDSGYPSEKRGELDDPEPREHGHSYSNRKYESDEDSLGSSGRVCVEKWNLLNSSRLHLPRASAVALEVQRLNALDLEKKIGKSILGKVHLAMVRYHEGGRFCEKGEEWDQESAVFHLEHAANLGELEAIVGLGLMYSQLPHHILADVSLKETEENKTKGFDYLLKAAEAGDRQSMILVARAFDSGQNLSPDRCQDWLEALHWYNTALEMTDCDEGGEYDGMQDEPRYMMLAREAEMLFTGGYGLEKDPQRSGDLYTQAAEAAMEAMKGRLANQYYQKAEEAWAQMEE';
    store.setDtaDrug(drug.value);
    store.setDtaProtein(protein.value);
    ElMessage.success('示例数据已加载');
};

const dtaRun = async () => {
    if (!drug.value || !protein.value) {
        ElMessage.warning('请输入药物SMILES和蛋白质序列');
        return;
    }
    loading.value = true;
    dtaResult.value = '';
    try {
        const response = await request.post('/predict_dta', {
            smiles: drug.value,
            protein: protein.value,
        });
        dtaResult.value = response.affinity;
        ElMessage.success('预测成功！');
    } catch (error) {
        console.error('请求失败:', error);
        ElMessage.error('预测失败，请检查输入或稍后重试');
    } finally {
        loading.value = false;
    }
};

const dtaReset = () => {
    drug.value = '';
    protein.value = '';
    dtaResult.value = '';
    store.setDtaDrug('');
    store.setDtaProtein('');
};
</script>
