<template>
    <div class="titleContainer">
        <div class="titleIcon">
            <img class="titleIconImg" src="../../assets//images/indexLogo.png" alt="Title Icon"></img>
            <div class="titleText">药物预测分析平台</div>
        </div>
        <div class="titleBtnList">
            <!-- 使用 v-for 渲染 TitleBtn 组件 -->
            <TitleBtn v-for="(item, index) in menuRouter" :key="index" :title="item.title"
                @click="titleRoute(item.path)" />
            <TitleBtn title="功能" @click="handleDropDown" :hasDropdown="true">
                <template #dropdown>
                    <div class="dropdown-menu">
                        <div class="menu-item" @click="titleRoute('data')">DTA</div>
                        <div class="menu-item" @click="titleRoute('ADC')">ADC</div>
                        <div class="menu-item" @click="titleRoute('antigen')">抗原抗体亲和力预测</div>
                    </div>
                </template>
            </TitleBtn>
            <TitleBtn title="登录/退出" @click="handleDropDown" :hasDropdown="true">
                <template #dropdown>
                    <div class="dropdown-menu2">
                        <div class="menu-item" @click="titleRoute('login')">登录</div>
                        <div class="menu-item" @click="TitlelogOut">退出</div>
                    </div>
                </template>
            </TitleBtn>


        </div>
    </div>
</template>


<script setup>
import { useRouter } from 'vue-router';
import TitleBtn from '../../components/titleBtn.vue';
import { constantRoute } from '../../router/router';
import { ref } from 'vue';
import { useUserStore } from '../../store/user';  // Ensure this is correct

const menuRouter = constantRoute[0].children
    .filter(child => ['home', 'contact'].includes(child.name)) // 只筛选 home, login, contact
    .map(child => ({
        title: child.meta.title,
        path: `/${child.path}`,
    }));

const router = useRouter();
const changeDropdown = ref(false);

// Initialize the userStore
const userStore = useUserStore();  // Initialize the userStore here

const titleRoute = (path) => {
    router.push({ path: `${path}` });
};

const TitlelogOut = () => {
    userStore.clearUserData();  // Clear user data in the store
    router.push('/login');  // Redirect to login page
};

const handleDropDown = () => {
    changeDropdown.value = !changeDropdown.value;
};
</script>

<style scoped>
.titleContainer {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
}

.titleIcon {
    width: 60%;
    display: flex;
    align-items: center;

    .titleIconImg {
        width: 7%;
        height: 7%;
    }

    .titleText {
        font-size: 1.5rem;
        font-weight: bold;
        margin-left: 1rem;
    }
}

.titleBtnList {
    height: 100%;
    width: 60%;
    display: flex;
    justify-content: space-between;
    position: relative;
    /* 为下拉菜单定位提供参考 */
}

/* 下拉菜单样式 */
.dropdown-menu {
    position: absolute;
    top: 100%;
    /* right: 0; */
    left: 50%;
    /* 让菜单靠右对齐 */
    width: 177px;
    /* 宽度 */
    /* 背景色 */
    z-index: 1000;
    /* 确保在下拉菜单最上层 */
    padding: 0 !important;
}

.dropdown-menu2 {
    position: absolute;
    top: 100%;
    /* right: 0; */
    left: 200%;
    /* 让菜单靠右对齐 */
    width: 150px;
    /* 宽度 */
    /* 背景色 */
    z-index: 1000;
    /* 确保在下拉菜单最上层 */
    padding: 0 !important;
}

/* 菜单项样式 */
.menu-item {
    width: 100%;
    height: 100px;
    /* 高度 */
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #fff;
    color: black;
    /* 文字颜色 */
    cursor: pointer;
    /* border-bottom: 0.5px solid black; */
}

.menu-item:hover {
    background-color: #7fceef;
    color: white;
    /* 鼠标悬停时的背景色 */
}
</style>
