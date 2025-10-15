<!-- 注册页面 -->
<template>
    <div class="registerContain">
        <div class="registerCard">
            <div class="registerCardTitle">注册用户</div>
            <div class="registerAccount">
                <input v-model="form.account" type="text" class="register-input" placeholder="请输入账号或邮箱" name="account">
            </div>
            <div class="registerPhone">
                <input v-model="form.phone" type="text" class="register-input" placeholder="手机号" name="account">
            </div>
            <div class="registerPassword">
                <input v-model="form.password" type="password" class="register-input" placeholder="请输入密码"
                    name="password">
            </div>
            <div class="registerLogin" @click="handleRegister">注册</div>
            <div class="registerCreate" @click="backLogin">返回登录</div>
        </div>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router';
import { ref } from 'vue';
import { ElMessage } from 'element-plus';

const router = useRouter();

const form = ref({
    account: '',
    username: '',
    password: ''
});


// 注册逻辑
const handleRegister = () => {
    ElMessage.success('注册成功');
    form.value.account = '';
    form.value.phone = '';
    form.value.password = '';
};

const backLogin = () => {
    router.push('/login'); // 跳转到注册页面
};
</script>

<style lang="scss" scoped>
.registerContain {
    width: 100%;
    height: 120vh;
    display: flex;
    justify-content: center;
    align-items: center;
    // background-image: linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%);
    background-image: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);

    .registerCard {
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

        .registerCardTitle {
            width: 100%;
            height: 20%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 30px;
            font-weight: 800;
        }

        .registerAccount,
        .registerPassword,
        .registerPhone {
            width: 100%;
            height: 20%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 14px;
            color: #333;
            margin-top: -20px;
        }

        .register-input {
            width: 300px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            border: none;
            background-color: #f6f9f9;
            height: 40px;
        }

        .register-input:focus {
            outline: none;
            border-color: #409eff;
            box-shadow: 0 0 3px rgba(64, 158, 255, 0.3);
        }

        .registerLogin {
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

        .registerCreate {
            margin-left: 40%;
            color: gray;
            cursor: pointer;
        }

        .registerCreate:hover {
            color: #7fceef;
            /* 鼠标悬停时颜色变为深蓝色 */
        }
    }
}
</style>