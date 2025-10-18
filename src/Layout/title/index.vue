<template>
    <div class="titleContainer" ref="navContainer">
        <div class="titleIcon" @click="titleRoute('/home')">
            <img class="titleIconImg" src="../../assets/images/indexLogo.png" alt="Title Icon">
            <div class="titleText">药物预测分析平台</div>
        </div>
        <div class="titleBtnList">
            <TitleBtn v-for="item in menuItems" :key="item.title" :title="item.title" :has-dropdown="!!item.children"
                @main-click="handleMenuClick(item)">
                <template #dropdown v-if="item.children">
                    <div class="dropdown-menu">
                        <div class="menu-item" v-for="child in item.children" :key="child.title"
                            @click.stop="handleSubMenuClick(child)">
                            {{ child.title }}
                        </div>
                    </div>
                </template>
            </TitleBtn>
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

onMounted(() => {
    gsap.from(navContainer.value, {
        y: -80,
        opacity: 0,
        duration: 0.8,
        ease: 'power3.out'
    });
    gsap.from('.titleBtn', {
        y: -20,
        opacity: 0,
        duration: 0.5,
        stagger: 0.1,
        delay: 0.5,
        ease: 'power2.out'
    });
});
</script>

<style scoped>
.titleContainer {
    width: 100%;
    height: 70px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 40px;
    background-color: #ffffff;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1000;
}

.titleIcon {
    display: flex;
    align-items: center;
    cursor: pointer;
}

.titleIconImg {
    width: 45px;
    height: 45px;
    margin-right: 15px;
}

.titleText {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e3a8a;
    /* 主题蓝色 */
}

.titleBtnList {
    display: flex;
    align-items: center;
    height: 100%;
}

.dropdown-menu {
    position: absolute;
    top: calc(100% + 5px);
    left: 50%;
    transform: translateX(-50%);
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    padding: 10px 0;
    min-width: 180px;
    overflow: hidden;
    visibility: hidden;
    /* 由gsap控制 */
    opacity: 0;
    /* 由gsap控制 */
}

.menu-item {
    padding: 12px 20px;
    font-size: 1rem;
    color: #333;
    cursor: pointer;
    transition: background-color 0.3s ease, color 0.3s ease;
    text-align: center;
}

.menu-item:hover {
    background-color: #f0f5ff;
    /* 悬停淡蓝色 */
    color: #1e3a8a;
    /* 主题蓝色 */
}
</style>
