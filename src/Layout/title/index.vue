<template>
    <div class="titleContainer" ref="navContainer">
        <!-- 背景装饰 -->
        <div class="nav-decoration"></div>

        <!-- Logo区域 -->
        <div class="titleIcon" @click="titleRoute('/home')">
            <div class="logo-container">
                <div class="logo-wrapper">
                    <img class="titleIconImg" src="../../assets/images/indexLogo.png" alt="Title Icon">
                    <div class="logo-glow"></div>
                </div>
                <div class="brand-info">
                    <div class="titleText">智药慧测</div>
                    <div class="brand-subtitle">AI Medicine Analysis</div>
                </div>
            </div>
        </div>

        <!-- 导航菜单 -->
        <div class="titleBtnList">
            <TitleBtn v-for="item in menuItems" :key="item.title" :title="item.title" :has-dropdown="!!item.children"
                @main-click="handleMenuClick(item)">
                <template #dropdown v-if="item.children">
                    <div class="dropdown-menu">
                        <div class="dropdown-header">
                            <span class="dropdown-title">{{ item.title }}</span>
                        </div>
                        <div class="menu-items-container">
                            <div class="menu-item" v-for="child in item.children" :key="child.title"
                                @click.stop="handleSubMenuClick(child)">
                                <div class="menu-item-content">
                                    <i :class="getMenuIcon(child.title)" class="menu-icon"></i>
                                    <span class="menu-text">{{ child.title }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </template>
            </TitleBtn>
        </div>

        <!-- 用户信息区域 -->
        <div class="user-section" v-if="userStore.isLoggedIn">
            <div class="user-avatar">
                <i class="fas fa-user"></i>
            </div>
            <span class="username">{{ userStore.username }}</span>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import TitleBtn from '../../components/titleBtn.vue';
import { useUserStore } from '../../store/user';
import gsap from 'gsap';

const router = useRouter();
const userStore = useUserStore();
const navContainer = ref(null);

const menuItems = ref([
    { title: '主页', path: '/home' },
    {
        title: '功能',
        children: [
            { title: 'DTA', path: '/data' },
            { title: 'ADC', path: '/ADC' },
            { title: '抗原抗体亲和力预测', path: '/antigen' },
        ]
    },
    { title: '联系我们', path: '/contact' },
    {
        title: '登录/注册',
        children: [
            { title: '登录', path: '/login' },
            { title: '退出', action: 'logout' },
        ]
    }
]);

const titleRoute = (path) => {
    if (path) router.push(path);
};

const handleMenuClick = (item) => {
    if (item.path) {
        titleRoute(item.path);
    }
};

const handleSubMenuClick = (child) => {
    if (child.path) {
        titleRoute(child.path);
    } else if (child.action === 'logout') {
        userStore.clearUserData();
        router.push('/login');
    }
};

const getMenuIcon = (title) => {
    const iconMap = {
        'DTA': 'fas fa-dna',
        'ADC': 'fas fa-microscope',
        '抗原抗体亲和力预测': 'fas fa-virus',
        '登录': 'fas fa-sign-in-alt',
        '退出': 'fas fa-sign-out-alt'
    };
    return iconMap[title] || 'fas fa-circle';
};

onMounted(() => {
    // 导航栏入场动画
    gsap.from(navContainer.value, {
        y: -100,
        opacity: 0,
        duration: 1,
        ease: 'power3.out'
    });

    // 延迟执行以确保DOM完全渲染
    setTimeout(() => {
        gsap.from('.titleBtn', {
            //这里设置导致了错误
            opacity: 0,
            duration: 0.6,
            stagger: 0.1,
            ease: 'power2.out'
        });
    }, 100);

    gsap.from('.logo-container', {
        x: -100,
        opacity: 0,
        duration: 0.8,
        delay: 0.3,
        ease: 'power2.out'
    });

    // 滚动监听 - 添加节流优化
    let ticking = false;
    const handleScroll = () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                const scrolled = window.scrollY > 50;
                const currentlyScrolled = navContainer.value?.classList.contains('scrolled');

                if (scrolled && !currentlyScrolled) {
                    navContainer.value?.classList.add('scrolled');
                } else if (!scrolled && currentlyScrolled) {
                    navContainer.value?.classList.remove('scrolled');
                }
                ticking = false;
            });
            ticking = true;
        }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });

    // 组件卸载时清理事件监听
    return () => {
        window.removeEventListener('scroll', handleScroll);
    };
});
</script>

<style scoped>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

.titleContainer {
    width: 100%;
    height: 80px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 50px;
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.98) 0%, rgba(59, 130, 246, 0.98) 50%, rgba(96, 165, 250, 0.98) 100%);
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(30, 64, 175, 0.3);
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);

    position: fixed;
    top: 0;
    left: 0;
    z-index: 1000;
    overflow: visible;
}

/* 背景装饰 */
.nav-decoration {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100"><defs><pattern id="smallGrid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23smallGrid)"/></svg>');
    opacity: 0.3;
    pointer-events: none;
}

/* Logo区域 */
.titleIcon {
    cursor: pointer;
    transition: transform 0.3s ease;
}

.titleIcon:hover {
    transform: translateY(-2px);
}

.logo-container {
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo-wrapper {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}

.titleIconImg {
    width: 50px;
    height: 50px;
    border-radius: 12px;
    transition: all 0.3s ease;
    position: relative;
    z-index: 2;
}

.logo-glow {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.4) 0%, transparent 70%);
    border-radius: 12px;
    opacity: 0;
    transition: opacity 0.3s ease;
    transform: scale(1.2);
}

.titleIcon:hover .logo-glow {
    opacity: 1;
}

.brand-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.titleText {
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    background: linear-gradient(45deg, #ffffff, #e0f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.brand-subtitle {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.8);
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* 导航菜单 */
.titleBtnList {
    display: flex;
    align-items: center;
    height: 100%;
    gap: 5px;
    position: relative;
    z-index: 10;
    flex-shrink: 0;
}

/* 确保导航按钮可见和对齐的样式 */
.titleBtnList :deep(.titleBtn) {
    color: white !important;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    min-width: 80px;
    height: 50px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    white-space: nowrap;
    flex-shrink: 0;
    vertical-align: middle;
    line-height: 1;
}

/* 下拉菜单样式 */
.dropdown-menu {
    position: absolute;
    top: calc(100% + 10px);
    right: 0;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(30, 64, 175, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
    min-width: 220px;
    max-width: 280px;
    overflow: hidden;
    visibility: hidden;
    opacity: 0;
    transform: translateY(-10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.dropdown-header {
    padding: 15px 20px 10px;
    border-bottom: 1px solid rgba(30, 64, 175, 0.1);
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.05), rgba(59, 130, 246, 0.05));
}

.dropdown-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #1e40af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.menu-items-container {
    padding: 10px 0;
}

.menu-item {
    padding: 0;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
}

.menu-item-content {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    transition: all 0.3s ease;
}

.menu-icon {
    font-size: 1rem;
    color: #3b82f6;
    width: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.menu-text {
    font-size: 1rem;
    color: #374151;
    font-weight: 500;
    transition: color 0.3s ease;
}

.menu-item:hover {
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.08), rgba(59, 130, 246, 0.08));
}

.menu-item:hover .menu-icon {
    color: #1e40af;
    transform: scale(1.1);
}

.menu-item:hover .menu-text {
    color: #1e40af;
}

.menu-item::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 3px;
    height: 0;
    background: linear-gradient(to bottom, #3b82f6, #60a5fa);
    transition: height 0.3s ease;
    border-radius: 0 2px 2px 0;
}

.menu-item:hover::before {
    height: 70%;
}

/* 用户信息区域 */
.user-section {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(255, 255, 255, 0.1);
    padding: 8px 16px;
    border-radius: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.user-section:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
}

.user-avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(45deg, #fbbf24, #f59e0b);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.9rem;
    box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
}

.username {
    color: white;
    font-weight: 600;
    font-size: 0.95rem;
}

/* 响应式设计 */
@media (max-width: 1200px) {
    .titleContainer {
        padding: 0 30px;
    }

    .titleBtnList :deep(.titleBtn) {
        padding: 0 15px;
        font-size: 1rem;
        min-width: 70px;
        height: 50px !important;
    }

    .dropdown-menu {
        min-width: 200px;
        right: -20px;
    }
}

@media (max-width: 768px) {
    .titleContainer {
        padding: 0 20px;
        height: 70px;
    }

    .titleText {
        font-size: 1.4rem;
    }

    .brand-subtitle {
        display: none;
    }

    .user-section {
        display: none;
    }

    .titleBtnList :deep(.titleBtn) {
        padding: 0 12px;
        font-size: 0.9rem;
        min-width: 60px;
        height: 45px !important;
    }

    .dropdown-menu {
        min-width: 180px;
        right: -30px;
    }
}

/* 滚动时的样式变化 */
.titleContainer.scrolled {
    height: 70px;
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.95) 0%, rgba(59, 130, 246, 0.95) 50%, rgba(96, 165, 250, 0.95) 100%);
    box-shadow: 0 4px 20px rgba(30, 64, 175, 0.4);
    backdrop-filter: blur(25px);
}

.titleContainer.scrolled .titleIconImg {
    width: 45px;
    height: 45px;
}

.titleContainer.scrolled .titleText {
    font-size: 1.6rem;
}

.titleContainer.scrolled .brand-subtitle {
    opacity: 0.6;
}
</style>
