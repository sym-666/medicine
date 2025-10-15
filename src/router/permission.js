import router from "@/router";
import { useUserStore } from "@/store/user";
import { ElMessage } from "element-plus";

const whiteList = ["/home", "/login", "/contact", "/register"]; // 登录页和注册页不受限制

router.beforeEach((to, from, next) => {
  const userStore = useUserStore();
  const token = userStore.token;

  if (token) {
    next(); // 已登录，放行
  } else {
    if (whiteList.includes(to.path)) {
      next(); // 在白名单内，放行
    } else {
      ElMessage.warning("请先登录");
      next("/login"); // 重定向到登录页
    }
  }
});
