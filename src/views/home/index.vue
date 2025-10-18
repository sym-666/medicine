<template>
    <div class="home-container">
        <!-- Hero Section -->
        <section class="hero-section">
            <div class="hero-content">
                <h1 class="hero-title">智能药物分析平台</h1>
                <p class="hero-subtitle">
                    利用前沿AI技术，加速您的药物研发进程，精准预测与分析。
                </p>
                <button class="hero-button" @click="goToData">开始分析</button>
            </div>
        </section>

        <!-- Features Section -->
        <section class="features-section">
            <h2 class="section-title">平台核心功能</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3 class="feature-title">AI驱动的预测</h3>
                    <p class="feature-description">
                        基于深度学习模型，提供高精度的药物靶点和活性预测。
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <h3 class="feature-title">多维数据分析</h3>
                    <p class="feature-description">
                        整合并分析来自不同来源的复杂数据集，发现潜在关联。
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-flask"></i>
                    </div>
                    <h3 class="feature-title">虚拟筛选</h3>
                    <p class="feature-description">
                        快速筛选大规模化合物库，高效发现潜在的候选药物。
                    </p>
                </div>
            </div>
        </section>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router';

const router = useRouter();

const goToData = () => {
    router.push('/data');
};
</script>

<style scoped>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

.home-container {
    width: 100%;
    overflow-x: hidden;
    background-color: #f8f9fa;
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

/* Hero Section */
.hero-section {
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 100px 20px;
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    /* background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); */
    background-image: url("../../assets/images/background.jpg");
    color: white;
    min-height: 60vh;
}

.hero-content {
    max-width: 800px;
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
    font-size: 3.5rem;
    font-weight: bold;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.hero-subtitle {
    font-size: 1.5rem;
    margin-bottom: 40px;
    opacity: 0.9;
}

.hero-button {
    padding: 15px 35px;
    font-size: 1.2rem;
    font-weight: bold;
    color: #1e3a8a;
    background-color: white;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.hero-button:hover {
    transform: translateY(-3px);
    background-color: #f0f0f0;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* Features Section */
.features-section {
    padding: 80px 20px;
    text-align: center;
}

.section-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1e3a8a;
    margin-bottom: 60px;
    position: relative;
}

.section-title::after {
    content: '';
    display: block;
    width: 80px;
    height: 4px;
    background-color: #3b82f6;
    margin: 20px auto 0;
    border-radius: 2px;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 40px;
    max-width: 1200px;
    margin: 0 auto;
}

.feature-card {
    background-color: white;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: popIn 0.5s ease-in-out forwards;
    opacity: 0;
}

.feature-card:nth-child(1) {
    animation-delay: 0.2s;
}

.feature-card:nth-child(2) {
    animation-delay: 0.4s;
}

.feature-card:nth-child(3) {
    animation-delay: 0.6s;
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

.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
}

.feature-icon {
    font-size: 3rem;
    color: #3b82f6;
    margin-bottom: 20px;
}

.feature-title {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1e3a8a;
    margin-bottom: 15px;
}

.feature-description {
    font-size: 1rem;
    color: #555;
    line-height: 1.6;
}
</style>
