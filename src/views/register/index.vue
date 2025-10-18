<template>
    <div class="auth-container">
        <div class="auth-card">
            <h1 class="auth-title">创建新账户</h1>
            <p class="auth-subtitle">加入我们，开启智能药物分析之旅</p>

            <el-form ref="registerFormRef" :model="registerForm" :rules="registerRules" class="auth-form"
                @keyup.enter="handleRegister">
                <el-form-item prop="account">
                    <el-input v-model="registerForm.account" placeholder="请输入您的邮箱作为账号" size="large" clearable>
                        <template #prefix>
                            <i class="fas fa-envelope"></i>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="phone">
                    <el-input v-model="registerForm.phone" placeholder="请输入手机号" size="large" clearable>
                        <template #prefix>
                            <i class="fas fa-mobile-alt"></i>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="password">
                    <el-input v-model="registerForm.password" type="password" placeholder="请设置密码" size="large"
                        show-password clearable>
                        <template #prefix>
                            <i class="fas fa-lock"></i>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="confirmPassword">
                    <el-input v-model="registerForm.confirmPassword" type="password" placeholder="请再次输入密码" size="large"
                        show-password clearable>
                        <template #prefix>
                            <i class="fas fa-check-circle"></i>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item>
                    <el-button type="primary" class="auth-button" size="large" @click="handleRegister"
                        :loading="loading">
                        注 册
                    </el-button>
                </el-form-item>
            </el-form>

            <div class="auth-footer">
                <span>已有账户？</span>
                <el-link type="primary" @click="backToLogin">返回登录</el-link>
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
const registerFormRef = ref(null);
const loading = ref(false);

const registerForm = reactive({
    account: '',
    phone: '',
    password: '',
    confirmPassword: '',
});

const validatePass = (rule, value, callback) => {
    if (value === '') {
        callback(new Error('请再次输入密码'));
    } else if (value !== registerForm.password) {
        callback(new Error("两次输入的密码不一致"));
    } else {
        callback();
    }
};

const registerRules = reactive({
    account: [
        { required: true, message: '请输入邮箱地址', trigger: 'blur' },
        { type: 'email', message: '请输入有效的邮箱地址', trigger: ['blur', 'change'] }
    ],
    phone: [
        { required: true, message: '请输入手机号', trigger: 'blur' },
        { pattern: /^1[3-9]\d{9}$/, message: '请输入有效的手机号', trigger: 'blur' }
    ],
    password: [
        { required: true, message: '请输入密码', trigger: 'blur' },
        { min: 6, message: '密码长度不能少于6位', trigger: 'blur' }
    ],
    confirmPassword: [
        { required: true, validator: validatePass, trigger: 'blur' }
    ],
});

const handleRegister = async () => {
    if (!registerFormRef.value) return;
    await registerFormRef.value.validate(async (valid) => {
        if (valid) {
            loading.value = true;
            try {
                await userStore.register(registerForm);
                ElMessage.success('注册成功！即将跳转到登录页');
                setTimeout(() => {
                    router.push('/login');
                }, 1500);
            } catch (error) {
                ElMessage.error(error.message || '注册失败，请重试');
            } finally {
                loading.value = false;
            }
        }
    });
};

const backToLogin = () => {
    router.push('/login');
};
</script>