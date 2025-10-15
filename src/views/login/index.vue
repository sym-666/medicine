<template>
    <!-- 登录页面 -->
    <div class="loginContain">
        <div class="loginCard">
            <div class="loginCardTitle">药物预测分析平台</div>
            <div class="loginAccount">
                <input v-model="account" type="text" class="login-input" placeholder="请输入账号或邮箱" name="account">

            </div>
            <div class="loginPassword">
                <input v-model="password" type="password" class="login-input" placeholder="请输入密码" name="password">
            </div>
            <div class="loginLogin" @click="handleLogin">登录</div>
            <div class="loginCreate" @click="goToRegister">注册</div>
        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
// import { useUserStore } from '@/store/user';
import { useUserStore } from '@/store/user/index.js';
import { ElMessage } from 'element-plus';

const router = useRouter();
const userStore = useUserStore();

const account = ref('');
const password = ref('');

const handleLogin = async () => {
    try {
        const success = await userStore.login(account.value, password.value);
        if (success) {
            ElMessage.success('登录成功');
            router.push('/home'); // 登录后跳转首页
        }
    } catch (error) {
        ElMessage.error(error.message);
    }
};

const goToRegister = () => {
    router.push('/register'); // 跳转到注册页面
};
</script>


<style lang="scss" scoped>
.loginContain {
    width: 100%;
    height: 120vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background-image: linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%);

    .loginCard {
        width: 30%;
        height: 500px;
        margin-top: -200px;
        border-radius: 10px;
        background-color: #ffffff;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;

        .loginCardTitle {
            width: 100%;
            height: 20%;
            // background-color: #1a355e;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 30px;
            font-weight: 800;

        }

        .loginAccount,
        .loginPassword {
            width: 100%;
            height: 20%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 14px;
            color: #333;
            margin-top: -20px;
            // background-color: red;
        }

        .login-input {
            width: 300px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            border: none;
            background-color: #f6f9f9;
            height: 40px;
        }

        .login-input:focus {
            outline: none;
            border-color: #409eff;
            box-shadow: 0 0 3px rgba(64, 158, 255, 0.3);
        }

        .loginLogin {
            width: 50%;
            height: 10%;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #7fceef;
            color: white;
            border-radius: 3%;
            cursor: pointer;

            &:hover {
                background-color: #a1e2f6;
            }
        }

        .loginCreate {
            margin-left: 40%;
            color: gray;
            cursor: pointer;
        }

        .loginCreate:hover {
            color: #7fceef;
            /* 鼠标悬停时颜色变为深蓝色 */
        }
    }
}
</style>
