import { defineStore } from "pinia";
import request from "@/utils/request"; // 引入axios实例

// 模拟用户数据库
const getMockUsers = () => {
  const users = localStorage.getItem("users");
  // 如果没有用户，则初始化一个包含admin的数组
  if (!users) {
    const adminUser = [
      { account: "admin", password: "123456", phone: "12345678900" },
    ];
    localStorage.setItem("users", JSON.stringify(adminUser));
    return adminUser;
  }
  return JSON.parse(users);
};

export const useUserStore = defineStore("user", {
  state: () => ({
    token: localStorage.getItem("token") || "",
    userInfo: JSON.parse(localStorage.getItem("userInfo")) || null,
    users: getMockUsers(), // 模拟的用户数据表
  }),
  actions: {
    // 注册方法
    async register(form) {
      // 模拟API调用
      return new Promise((resolve, reject) => {
        setTimeout(() => {
          const userExists = this.users.some(
            (user) => user.account === form.account
          );
          if (userExists) {
            reject(new Error("该邮箱已被注册"));
          } else {
            const newUser = { ...form };
            delete newUser.confirmPassword; // 不需要存储确认密码
            this.users.push(newUser);
            localStorage.setItem("users", JSON.stringify(this.users));
            resolve(true);
          }
        }, 500);
      });
    },

    // 登录方法
    async login(account, password) {
      // 模拟API调用
      return new Promise((resolve, reject) => {
        setTimeout(() => {
          const user = this.users.find(
            (user) => user.account === account && user.password === password
          );

          if (user) {
            const token = "mock-token-" + new Date().getTime();
            const userInfo = { account: user.account, phone: user.phone };

            localStorage.setItem("token", token);
            localStorage.setItem("userInfo", JSON.stringify(userInfo));

            this.token = token;
            this.userInfo = userInfo;

            resolve(true);
          } else {
            reject(new Error("账号或密码错误"));
          }
        }, 500);
      });
    },

    // 退出登录
    logout() {
      this.token = "";
      this.userInfo = null;
      localStorage.removeItem("token");
      localStorage.removeItem("userInfo");
    },
  },
});
