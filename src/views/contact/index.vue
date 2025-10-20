<template>
    <div class="contact-container">
        <!-- Hero Section -->
        <section class="contact-hero">
            <div class="hero-content">
                <h1 class="hero-title">联系我们</h1>
                <p class="hero-subtitle">Connect with Our Team</p>
                <p class="hero-description">
                    我们致力于为您提供最专业的AI药物分析解决方案
                </p>
            </div>
        </section>

        <!-- Contact Content -->
        <section class="contact-content">
            <div class="container">
                <div class="contact-grid">
                    <!-- Contact Information -->
                    <div class="contact-info-section">
                        <h2 class="section-title">联系信息</h2>
                        <div class="contact-cards">
                            <div class="contact-card">
                                <div class="contact-icon">
                                    <i class="fas fa-user"></i>
                                </div>
                                <div class="contact-details">
                                    <h3 class="contact-label">联系人</h3>
                                    <p class="contact-value">袁立沛</p>
                                </div>
                            </div>

                            <div class="contact-card">
                                <div class="contact-icon">
                                    <i class="fas fa-phone"></i>
                                </div>
                                <div class="contact-details">
                                    <h3 class="contact-label">电话/微信</h3>
                                    <p class="contact-value">18612506617</p>
                                </div>
                            </div>

                            <div class="contact-card">
                                <div class="contact-icon">
                                    <i class="fas fa-envelope"></i>
                                </div>
                                <div class="contact-details">
                                    <h3 class="contact-label">邮箱地址</h3>
                                    <p class="contact-value">yuanlp2004@163.com</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="contact-info-section">
                        <h2 class="section-title">联系信息</h2>
                        <div class="contact-cards">
                            <div class="contact-card">
                                <div class="contact-icon">
                                    <i class="fas fa-user"></i>
                                </div>
                                <div class="contact-details">
                                    <h3 class="contact-label">联系人</h3>
                                    <p class="contact-value">陆京超</p>
                                </div>
                            </div>

                            <div class="contact-card">
                                <div class="contact-icon">
                                    <i class="fas fa-phone"></i>
                                </div>
                                <div class="contact-details">
                                    <h3 class="contact-label">电话/微信</h3>
                                    <p class="contact-value">15927019964</p>
                                </div>
                            </div>

                            <div class="contact-card">
                                <div class="contact-icon">
                                    <i class="fas fa-envelope"></i>
                                </div>
                                <div class="contact-details">
                                    <h3 class="contact-label">邮箱地址</h3>
                                    <p class="contact-value">1368726945@qq.com</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Platform Information -->
                    
                </div>
            </div>
        </section>

        <!-- Contact Form Section -->
        <section class="contact-form-section">
            <div class="container">
                <div class="form-wrapper">
                    <h2 class="form-title">发送消息</h2>
                    <p class="form-subtitle">有任何问题或建议，请随时与我们联系</p>

                    <form class="contact-form" @submit.prevent="submitForm">
                        <div class="form-row">
                            <div class="form-group">
                                <label for="name">姓名</label>
                                <input type="text" id="name" v-model="form.name" placeholder="请输入您的姓名" required>
                            </div>
                            <div class="form-group">
                                <label for="email">邮箱</label>
                                <input type="email" id="email" v-model="form.email" placeholder="请输入您的邮箱" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label for="subject">主题</label>
                            <input type="text" id="subject" v-model="form.subject" placeholder="请输入消息主题" required>
                        </div>

                        <div class="form-group">
                            <label for="message">消息内容</label>
                            <textarea id="message" v-model="form.message" placeholder="请输入您的消息内容..." rows="5"
                                required></textarea>
                        </div>

                        <button type="submit" class="submit-button">
                            <i class="fas fa-paper-plane"></i>
                            发送消息
                        </button>
                    </form>
                </div>
            </div>
        </section>
    </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();

// 表单数据
const form = ref({
    name: '',
    email: '',
    subject: '',
    message: ''
});

// 开始分析功能
const startAnalysis = () => {
    router.push('/data');
};

// 提交表单
const submitForm = () => {
    // 这里可以添加表单提交逻辑
    console.log('Form submitted:', form.value);

    // 显示成功消息（可以使用Element Plus的消息组件）
    alert('消息发送成功！我们会尽快回复您。');

    // 重置表单
    form.value = {
        name: '',
        email: '',
        subject: '',
        message: ''
    };
};
</script>


<style scoped>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

.contact-container {
    width: 100%;
    min-height: 100vh;
    background: linear-gradient(to bottom, #f8fafe, #ffffff);
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
.contact-hero {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
    color: white;
    padding: 120px 20px 80px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.contact-hero::before {
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
    font-size: 3.5rem;
    font-weight: 900;
    margin-bottom: 15px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.hero-subtitle {
    font-size: 1.5rem;
    margin-bottom: 20px;
    opacity: 0.9;
    font-weight: 600;
}

.hero-description {
    font-size: 1.2rem;
    margin-bottom: 40px;
    opacity: 0.85;
    line-height: 1.6;
}

/* Contact Content */
.contact-content {
    padding: 100px 0;
}

.contact-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 60px;
    align-items: start;
}

.section-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e40af;
    margin-bottom: 40px;
    position: relative;
}

.section-title::after {
    content: '';
    display: block;
    width: 80px;
    height: 4px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
    margin: 20px 0;
    border-radius: 2px;
}

/* Contact Cards */
.contact-cards {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.contact-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(30, 64, 175, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.1);
    display: flex;
    align-items: center;
    gap: 25px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.contact-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
}

.contact-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 50px rgba(30, 64, 175, 0.15);
}

.contact-icon {
    width: 70px;
    height: 70px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.8rem;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    flex-shrink: 0;
}

.contact-details {
    flex: 1;
}

.contact-label {
    font-size: 1rem;
    color: #64748b;
    margin: 0 0 8px 0;
    font-weight: 500;
}

.contact-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e40af;
    margin: 0;
}

/* Platform Info */
.platform-card {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-radius: 25px;
    padding: 40px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    box-shadow: 0 15px 50px rgba(30, 64, 175, 0.1);
}

.platform-header {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 30px;
}

.platform-logo {
    width: 80px;
    height: 80px;
    background: linear-gradient(45deg, #1e40af, #3b82f6);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 2.2rem;
    box-shadow: 0 10px 30px rgba(30, 64, 175, 0.3);
}

.platform-title h2 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e40af;
    margin: 0 0 8px 0;
}

.platform-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 500;
    margin: 0;
}

.platform-description h3 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 15px;
}

.platform-description p {
    font-size: 1.1rem;
    color: #64748b;
    margin-bottom: 25px;
    line-height: 1.6;
}

.platform-features {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 30px;
}

.feature-tag {
    background: rgba(59, 130, 246, 0.1);
    color: #1e40af;
    padding: 10px 20px;
    border-radius: 25px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.platform-cta {
    background: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(30, 64, 175, 0.1);
}

.cta-text {
    font-size: 1.1rem;
    color: #64748b;
    margin-bottom: 20px;
    line-height: 1.6;
}

.cta-button {
    background: linear-gradient(45deg, #1e40af, #3b82f6);
    color: white;
    padding: 15px 30px;
    border: none;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.3);
}

.cta-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(30, 64, 175, 0.4);
}

/* Contact Form */
.contact-form-section {
    padding: 100px 0;
    background: linear-gradient(135deg, #f8fafc, #e0f2fe);
}

.form-wrapper {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 50px;
    border-radius: 25px;
    box-shadow: 0 20px 60px rgba(30, 64, 175, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.1);
}

.form-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e40af;
    text-align: center;
    margin-bottom: 15px;
}

.form-subtitle {
    font-size: 1.1rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 40px;
    line-height: 1.6;
}

.contact-form {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.form-group label {
    font-weight: 600;
    color: #1e40af;
    font-size: 1rem;
}

.form-group input,
.form-group textarea {
    padding: 15px 20px;
    border: 2px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: #f8fafc;
}

.form-group input:focus,
.form-group textarea:focus {
    outline: none;
    border-color: #3b82f6;
    background: white;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-group textarea {
    resize: vertical;
    min-height: 120px;
}

.submit-button {
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
    justify-content: center;
    gap: 12px;
    margin-top: 20px;
    box-shadow: 0 8px 30px rgba(30, 64, 175, 0.3);
}

.submit-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(30, 64, 175, 0.4);
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }

    .contact-grid {
        grid-template-columns: 1fr;
        gap: 40px;
    }

    .contact-card {
        flex-direction: column;
        text-align: center;
        padding: 25px;
    }

    .form-row {
        grid-template-columns: 1fr;
        gap: 20px;
    }

    .form-wrapper {
        padding: 30px 20px;
        margin: 0 20px;
    }

    .platform-header {
        flex-direction: column;
        text-align: center;
    }
}
</style>
