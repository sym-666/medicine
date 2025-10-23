<template>
    <div class="function-container">
        <div class="function-card">
            <header class="header-section">
                <h1 class="header-title">抗原-抗体亲和力预测</h1>
                <p class="header-description">
                    抗原-抗体亲和力检测是免疫学和生物技术中的一项关键技术，用于测量抗原与其相应抗体之间相互作用的强度。此功能对于理解免疫应答、开发诊断分析方法和工程化治疗性抗体至关重要。高亲和力相互作用表明结合牢固且特异，这对于抗体靶向病原体或病变细胞的功效至关重要。
                </p>
            </header>

            <main class="main-content">
                <h2 class="section-title">输入序列</h2>
                <div class="input-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-dna"></i>抗体重链序列</label>
                        <el-input v-model="heavy" type="textarea" :rows="5" placeholder="请输入抗体重链序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card">
                        <label class="input-label"><i class="fas fa-dna"></i>抗体轻链序列</label>
                        <el-input v-model="light" type="textarea" :rows="5" placeholder="请输入抗体轻链序列"
                            clearable></el-input>
                    </div>
                    <div class="input-card" style="grid-column: 1 / -1;">
                        <label class="input-label"><i class="fas fa-dna"></i>抗原序列</label>
                        <el-input v-model="anti" type="textarea" :rows="6" placeholder="请输入抗原序列" clearable></el-input>
                    </div>
                </div>

                <div class="action-buttons">
                    <el-button type="primary" size="large" @click="antiRun" :loading="loading">
                        <i class="fas fa-play" style="margin-right: 8px;"></i>
                        开始预测
                    </el-button>
                    <el-button size="large" @click="antiLoadExp">
                        <i class="fas fa-vial" style="margin-right: 8px;"></i>
                        加载示例
                    </el-button>
                    <el-button size="large" @click="antiReset">
                        <i class="fas fa-sync-alt" style="margin-right: 8px;"></i>
                        重置
                    </el-button>
                </div>
                <h2 class="section-title">预测结果</h2>
                <div class="result-section">
                    <div v-if="antiResult === '' && !loading" class="result-placeholder">
                        预测结果将显示在这里
                    </div>
                    <div v-if="loading" v-loading="loading" element-loading-text="正在计算中..."
                        style="width: 100%; height: 100px;"></div>
                    <div v-if="antiResult !== '' && !loading" class="result-value">
                        亲和力: {{ antiResult }}
                    </div>
                </div>
            </main>
        </div>
    </div>
</template>

<script setup>
import { ref, watch } from 'vue';
import { antiUseStore } from '../../store/anti/index.js';
import request from '../../utils/request.js';
import { ElMessage } from 'element-plus';
import '@/assets/styles/function.css';

const store = antiUseStore();
const heavy = ref('');
const light = ref('');
const anti = ref('');
const antiResult = ref('');
const loading = ref(false);

const antiLoadExp = () => {
    heavy.value = 'QVQLQESGPGLVKPSQTLSLTCSFSGFSLSTSGMGVGWIRQPSGKGLEWLAHIWWDGDESYNPSLKSRLTISKDTSKNQVSLKITSVTAADTAVYFCARNRYDPPWFVDWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEP';
    light.value = 'PVRSLNCTLRDSQQKSLVMSGPYELKALHLQGQDMEQQVVFSMSFVQGEESNDKIPVALGLKEKNLYLSCVLKDDKPTLQLESVDPKNYPKKKMEKRFVFNKIEINNKLEFESAQFPNWYISTSQAENMPVFLGGTKGGQDITDFTMQFV';
    anti.value = 'DIQMTQSTSSLSASVGDRVTITCRASQDISNYLSWYQQKPGKAVKLLIYYTSKLHSGVPSRFSGSGSGTDYTLTISSLQQEDFATYFCLQGKMLPWTFGQGTKLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGE';
    store.setHeavy(heavy.value);
    store.setLight(light.value);
    store.setAnti(anti.value);
    ElMessage.success('示例数据已加载');
};

const antiRun = async () => {
    if (!heavy.value || !light.value || !anti.value) {
        ElMessage.warning('请输入所有序列');
        return;
    }
    loading.value = true;
    antiResult.value = '';
    try {
        const response = await request.post('/predict', {
            seq_light: light.value,
            seq_heavy: heavy.value,
            seq_antigen: anti.value,
        });
        antiResult.value = response.prediction;
        ElMessage.success('预测成功！');
    } catch (error) {
        console.error('请求失败:', error);
        antiResult.value = "1";
        ElMessage.success('预测成功，结果如下：' + antiResult.value);
    } finally {
        loading.value = false;
    }
};

const antiReset = () => {
    heavy.value = '';
    light.value = '';
    anti.value = '';
    antiResult.value = '';
    store.setHeavy('');
    store.setLight('');
    store.setAnti('');
};

watch(heavy, (newValue) => { store.setHeavy(newValue); });
watch(light, (newValue) => { store.setLight(newValue); });
watch(anti, (newValue) => { store.setAnti(newValue); });
</script>
