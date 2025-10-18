<template>
    <div class="auth-container">
        <div class="auth-card">
            <h1 class="auth-title">欢迎回来</h1>
            <p class="auth-subtitle">登录您的药物预测分析平台账户</p>

            <el-form ref="loginFormRef" :model="loginForm" :rules="loginRules" class="auth-form"
                @keyup.enter="handleLogin">
                <el-form-item prop="account">
                    <el-input v-model="loginForm.account" placeholder="请输入账号或邮箱" size="large" clearable>
                        <template #prefix>
                            <i class="fas fa-user"></i>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="password">
                    <el-input v-model="loginForm.password" type="password" placeholder="请输入密码" size="large"
                        show-password clearable>
                        <template #prefix>
                            <i class="fas fa-lock"></i>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item>
                    <el-button type="primary" class="auth-button" size="large" @click="handleLogin" :loading="loading">
                        登 录
                    </el-button>
                </el-form-item>
            </el-form>

            <div class="auth-footer">
                <span>还没有账户？</span>
                <el-link type="primary" @click="goToRegister">立即注册</el-link>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, reactive } from 'vue';
import { useRouter } from 'vue-router';
import { useUserStore } from '@/store/user';
import { ElMessage } from 'element-plus';
import '@/assets/styles/auth.css';

const router = useRouter();
const userStore = useUserStore();
const loginFormRef = ref(null);
const loading = ref(false);

const loginForm = reactive({
    account: '',
    password: '',
});

const loginRules = reactive({
    account: [{ required: true, message: '请输入您的账号', trigger: 'blur' }],
    password: [{ required: true, message: '请输入您的密码', trigger: 'blur' }],
});

const handleLogin = async () => {
    if (!loginFormRef.value) return;
    await loginFormRef.value.validate(async (valid) => {
        if (valid) {
      loading.value = true;
      try {
        const success = await userStore.login(loginForm.account, loginForm.password);
        if (success) {
          ElMessage.success('登录成功');
          router.push('/home');
        }
      } catch (error) {
        ElMessage.error(error.message || '登录失败，请重试');
      } finally {
        loading.value = false;
      }
    }
  });
};

const goToRegister = () => {
  router.push('/register');
};
</script>
