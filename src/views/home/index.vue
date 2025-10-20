<template>
    <div class="home-container">
        <section class="hero-section">
            <div class="hero-content">
                <h1 class="hero-title">智药慧测</h1>
                <p class="hero-subtitle">基于人工智能的医药分析预测平台</p>
                <p class="hero-description">
                    集成深度学习与图神经网络，提供全方位的药物分析解决方案
                </p>
                <div class="hero-buttons">
                    <button class="hero-button primary" @click="goToData">开始预测</button>
                    <button class="hero-button secondary" @click="learnMore">了解更多</button>
                </div>
            </div>
            <div class="hero-stats">
                <div class="stat-item">
                    <span class="stat-number">3+</span>
                    <span class="stat-label">AI模型</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">99%</span>
                    <span class="stat-label">预测精度</span>
                </div>
            </div>
        </section>

        <section class="core-features-section">
            <div class="container">
                <h2 class="second-section-title">平台核心优势</h2>
                <div class="core-features-grid">
                    <div class="core-feature-card feature-extraction">
                        <div class="feature-header">
                            <div class="feature-icon">
                                <i class="fas fa-dna"></i>
                            </div>
                            <h3>特征提取功能</h3>
                        </div>
                        <div class="feature-content">
                            <p class="feature-intro">利用深度学习模型对氨基酸序列进行精准特征提取</p>
                            <div class="feature-techs">
                                <div class="tech-item">
                                    <i class="fas fa-brain"></i>
                                    <div class="tech-info">
                                        <strong>ProteinBERT & ESM-2</strong>
                                        <span>深度挖掘生物序列特征</span>
                                    </div>
                                </div>
                                <div class="tech-item">
                                    <i class="fas fa-layer-group"></i>
                                    <div class="tech-info">
                                        <strong>卷积神经网络(CNN)</strong>
                                        <span>捕捉局部特征模式</span>
                                    </div>
                                </div>
                                <div class="tech-item">
                                    <i class="fas fa-network-wired"></i>
                                    <div class="tech-info">
                                        <strong>图神经网络(GNN)</strong>
                                        <span>处理分子图形结构</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Model Prediction -->
                    <div class="core-feature-card model-prediction">
                        <div class="feature-header">
                            <div class="feature-icon">
                                <i class="fas fa-microscope"></i>
                            </div>
                            <h3>模型预测功能</h3>
                        </div>
                        <div class="feature-content">
                            <p class="feature-intro">集成多个先进AI模型，实现精准的药物预测分析</p>
                            <div class="prediction-models">
                                <div class="model-item">
                                    <div class="model-badge">MVSF-AB</div>
                                    <div class="model-info">
                                        <strong>抗体-抗原结合亲和力预测</strong>
                                        <span>多视图序列特征融合技术，为抗体工程和疫苗设计提供关键数据支持</span>
                                    </div>
                                </div>
                                <div class="model-item">
                                    <div class="model-badge">ADCNet</div>
                                    <div class="model-info">
                                        <strong>抗体-药物偶联物活性预测</strong>
                                        <span>综合考虑ADC各组件信息，模拟体内行为，优化药物设计</span>
                                    </div>
                                </div>
                                <div class="model-item">
                                    <div class="model-badge">ViDTA</div>
                                    <div class="model-info">
                                        <strong>药物-靶点亲和力预测</strong>
                                        <span>引入虚拟节点和注意力机制，提升小分子药物研发成功率</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        <section class="tech-stack-section">
            <div class="container">
                <h2 class="section-title">技术架构</h2>
                <div class="tech-stack-grid">
                    <div class="tech-category">
                        <h4>深度学习</h4>
                        <div class="tech-tags">
                            <span class="tech-tag">ProteinBERT</span>
                            <span class="tech-tag">ESM-2</span>
                            <span class="tech-tag">Transformer</span>
                        </div>
                    </div>
                    <div class="tech-category">
                        <h4>神经网络</h4>
                        <div class="tech-tags">
                            <span class="tech-tag">CNN</span>
                            <span class="tech-tag">GNN</span>
                            <span class="tech-tag">Attention</span>
                        </div>
                    </div>
                    <div class="tech-category">
                        <h4>预测模型</h4>
                        <div class="tech-tags">
                            <span class="tech-tag">MVSF-AB</span>
                            <span class="tech-tag">ADCNet</span>
                            <span class="tech-tag">ViDTA</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        <section class="cta-section">
            <div class="container">
                <div class="cta-content">
                    <h2>开启您的AI药物分析之旅</h2>
                    <p>立即体验我们的先进预测模型，加速您的研发进程</p>
                    <div class="cta-buttons">
                        <button class="cta-button" @click="goToData">
                            <i class="fas fa-rocket"></i>
                            立即开始
                        </button>
                    </div>
                </div>
            </div>
        </section>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router';
import { onMounted } from 'vue';

const router = useRouter();

const goToData = () => {
    router.push('/data');
};

const learnMore = () => {
    // 滚动到核心功能部分
    const element = document.querySelector('.core-features-section');
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
};

// 在组件挂载后初始化观察器
onMounted(() => {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            } else {
                entry.target.classList.remove('fade-in');
            }
        });
    }, { threshold: 0.15 });

    // 观察所有需要淡入效果的元素
    document.querySelectorAll('.tech-tag, .tech-category, .core-feature-card').forEach((element) => {
        observer.observe(element);
    });
});
</script>

<style scoped>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

*{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
.home-container {
    width: 100%;
    overflow-x: hidden;
    background-color: #f8fafe;
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }

    to {
        opacity: 1;
    }
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Hero Section */
.hero-section {
    background-image: url(../../assets/images/background.jpg);
    background-size: cover;
    color: white;
    padding: 120px 20px 80px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 300"><defs><pattern id="smallGrid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23smallGrid)"/></svg>');
    opacity: 0.3;
}

.hero-content {
    max-width: 800px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
    animation: slideInUp 1s ease-in-out;
}

@keyframes slideInUp {
    from {
        transform: translateY(50px);
        opacity: 0;
    }

    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.hero-title {
    font-size: 4rem;
    font-weight: 900;
    margin-bottom: 15px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    background: linear-gradient(45deg, #ffffff, #e0f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.8rem;
    margin-bottom: 20px;
    opacity: 0.95;
    font-weight: 600;
}

.hero-description {
    font-size: 1.2rem;
    margin-bottom: 40px;
    opacity: 0.85;
    line-height: 1.6;
}

.hero-buttons {
    display: flex;
    gap: 20px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 60px;
}

.hero-button {
    padding: 15px 35px;
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.hero-button.primary {
    background: linear-gradient(45deg, #ffffff, #f0f9ff);
    color: #1e40af;
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
}

.hero-button.primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(255, 255, 255, 0.3);
    background: linear-gradient(45deg, #f8fafc, #e0f2fe);
}

.hero-button.secondary {
    background: transparent;
    color: white;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.hero-button.secondary:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.5);
    transform: translateY(-3px);
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 60px;
    flex-wrap: wrap;
    margin-top: 40px;
}

.stat-item {
    text-align: center;
    animation: popIn 0.6s ease-in-out forwards;
    opacity: 0;
}

.stat-item:nth-child(1) {
    animation-delay: 0.2s;
}

.stat-item:nth-child(2) {
    animation-delay: 0.7s;
}

.stat-item:nth-child(3) {
    animation-delay: 0.7s;
}

@keyframes popIn {
    from {
        transform: scale(0.8);
        opacity: 0;
    }

    to {
        transform: scale(1);
        opacity: 1;
    }
}

.stat-number {
    display: block;
    font-size: 2.5rem;
    font-weight: 900;
    color: #fbbf24;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}

.stat-label {
    display: block;
    font-size: 1rem;
    opacity: 0.9;
    margin-top: 5px;
}

/* Core Features Section */
.core-features-section {
    padding: 100px 0;
    background: linear-gradient(to bottom, #f8fafe, #ffffff);
}

.section-title {
    font-size: 3rem;
    font-weight: 800;
    color: #fcfcfe;
    text-align: center;
    margin-bottom: 80px;
    position: relative;
}

.section-title::after {
    content: '';
    display: block;
    width: 100px;
    height: 4px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
    margin: 25px auto 0;
    border-radius: 2px;
}
.second-section-title {
    font-size: 3rem;
    font-weight: 800;
    color: #1e40af;
    text-align: center;
    margin-bottom: 80px;
    position: relative;
}

.second-section-title::after {
    content: '';
    display: block;
    width: 200px;
    height: 4px;
    background: linear-gradient(45deg, #3b82f6, #8dbdf8);
    margin: 25px auto 0;
    border-radius: 2px;
}


.core-features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 50px;
    margin-top: 60px;
}

.core-feature-card {
    background: white;
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 10px 40px rgba(30, 64, 175, 0.08);
    transition: all 0.4s ease;
    border: 1px solid rgba(59, 130, 246, 0.1);
    position: relative;
    overflow: hidden;
}

.core-feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
}

.core-feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 60px rgba(30, 64, 175, 0.15);
}

.feature-header {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 30px;
}

.feature-icon {
    width: 70px;
    height: 70px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: white;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
}

.feature-header h3 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e40af;
    margin: 0;
}

.feature-intro {
    font-size: 1.1rem;
    color: #64748b;
    margin-bottom: 25px;
    line-height: 1.6;
}

.feature-techs {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.tech-item {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 15px;
    background: #f8fafc;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
}

.tech-item i {
    font-size: 1.5rem;
    color: #3b82f6;
    min-width: 24px;
}

.tech-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.tech-info strong {
    color: #1e40af;
    font-weight: 600;
}

.tech-info span {
    color: #64748b;
    font-size: 0.9rem;
}

.prediction-models {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.model-item {
    display: flex;
    align-items: flex-start;
    gap: 15px;
    padding: 20px;
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-radius: 15px;
    transition: all 0.3s ease;
}

.model-item:hover {
    transform: translateX(5px);
    box-shadow: 0 5px 20px rgba(59, 130, 246, 0.1);
}

.model-badge {
    background: linear-gradient(45deg, #1e40af, #3b82f6);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    min-width: fit-content;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.3);
}

.model-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.model-info strong {
    color: #1e40af;
    font-weight: 600;
    font-size: 1.1rem;
}

.model-info span {
    color: #64748b;
    line-height: 1.5;
    font-size: 0.95rem;
}

/* Technology Stack Section */
.tech-stack-section {
    padding: 80px 0;
    background: linear-gradient(135deg, #1e40af, #3b82f6);
    color: white;
}

.tech-stack-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 60px;
}



.tech-category h4 {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 25px;
    color: #fbbf24;
}

.tech-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
}

.tech-tag {
    background: rgba(107, 99, 99, 0.205);
    color: white;
    padding: 10px 20px;
    border-radius: 25px;
    font-weight: 500;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.5s ease;
    opacity: 0;
    transform: translateY(20px);
}

.tech-tag.fade-in {
    opacity: 1;
    transform: translateY(0);
}

.tech-tag:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-3px);
}

/* 为tech-category添加淡入效果 */
.tech-category {
    text-align: center;
    opacity: 0;
    transform: translateY(30px);
    transition: all 0.6s ease;
}

.tech-category.fade-in {
    opacity: 1;
    transform: translateY(0);
}

.tech-category:nth-child(1) {
    transition-delay: 0.2s;
}

.tech-category:nth-child(2) {
    transition-delay: 0.4s;
}

.tech-category:nth-child(3) {
    transition-delay: 0.6s;
}

/* 为core-feature-card添加淡入效果 */
.core-feature-card {
    background: white;
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 10px 40px rgba(30, 64, 175, 0.08);
    transition: all 0.4s ease;
    border: 1px solid rgba(59, 130, 246, 0.1);
    position: relative;
    overflow: hidden;
    opacity: 0;
    transform: translateY(30px);
}

.core-feature-card.fade-in {
    opacity: 1;
    transform: translateY(0);
}

/* CTA Section */
.cta-section {
    padding: 100px 0;
    background: linear-gradient(135deg, #f8fafc, #e0f2fe, #f0f9ff);
    text-align: center;
}

.cta-content h2 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e40af;
    margin-bottom: 20px;
}

.cta-content p {
    font-size: 1.2rem;
    color: #64748b;
    margin-bottom: 40px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.cta-button {
    background: linear-gradient(45deg, #1e40af, #3b82f6);
    color: white;
    padding: 18px 40px;
    border: none;
    border-radius: 15px;
    font-size: 1.2rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 8px 30px rgba(30, 64, 175, 0.3);
}

.cta-button:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(30, 64, 175, 0.4);
    background: linear-gradient(45deg, #1d4ed8, #2563eb);
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }

    .hero-subtitle {
        font-size: 1.4rem;
    }

    .core-features-grid {
        grid-template-columns: 1fr;
        gap: 30px;
    }

    .hero-stats {
        gap: 40px;
    }

    .tech-stack-grid {
        grid-template-columns: 1fr;
        gap: 30px;
    }

    .section-title {
        font-size: 2.2rem;
    }
}
</style>
