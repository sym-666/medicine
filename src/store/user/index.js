import { defineStore } from 'pinia';
import { useRouter } from 'vue-router';

export const useUserStore = defineStore('user', {
    state: () => ({
        token: localStorage.getItem('token') || '', // 令牌
        userInfo: JSON.parse(localStorage.getItem('userInfo')) || null, // 用户信息
    }),
    actions: {
        // 登录方法
        async login(username, password) {
            return new Promise((resolve, reject) => {
                setTimeout(() => {
                    if (username === 'admin' && password === '123456') {
                        const token = 'mock-token-123456';
                        const userInfo = { username: 'admin' };

                        localStorage.setItem('token', token);
                        localStorage.setItem('userInfo', JSON.stringify(userInfo));

                        this.token = token;
                        this.userInfo = userInfo;

                        resolve(true);
                    } else {
                        reject(new Error('账号或密码错误'));
                    }
                }, 1000); // 模拟 API 延迟
            });
        },
        clearUserData() {
            this.token = '';
            this.account = '';
            this.password = '';
        },

        // 退出登录
        logout() {
            this.token = '';
            this.account = '';
            this.userInfo = null;
            localStorage.removeItem('token');
            localStorage.removeItem('userInfo');
        },
    },
});
