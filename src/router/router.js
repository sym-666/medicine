export const constantRoute = [
  {
    path: "/",
    component: () => import("../Layout/index.vue"),
    name: "layout",
    meta: {
      title: "",
      hidden: false,
      icon: "",
    },
    redirect: "/home", // 设置默认首页为 home
    children: [
      {
        path: "home",
        component: () => import("../views/home/index.vue"),
        name: "home",
        meta: {
          title: "AI智能药物检测",
          hidden: false,
        },
      },
      {
        path: "data",
        component: () => import("../views/data/index.vue"),
        name: "data",
        meta: {
          hidden: false,
          title: "DTA",
        },
      },
      {
        path: "ADC",
        component: () => import("../views/ADC/index.vue"),
        name: "ADC",
        meta: {
          hidden: false,
          title: "ADC",
        },
      },
      {
        path: "antigen",
        component: () => import("../views/antigen/index.vue"),
        name: "antigen",
        meta: {
          hidden: false,
          title: "抗原抗体亲和力预测",
        },
      },
      {
        path: "login", // 子路由路径不需要加 '/'
        component: () => import("../views/login/index.vue"),
        name: "login",
        meta: {
          title: "登录",
          hidden: false,
        },
      },
      {
        path: "contact", // 子路由路径不需要加 '/'
        component: () => import("../views/contact/index.vue"),
        name: "contact",
        meta: {
          title: "联系我们",
          hidden: false,
        },
      },
      {
        path: "register", // 子路由路径不需要加 '/'
        component: () => import("../views/register/index.vue"),
        name: "register",
        meta: {
          title: "注册用户",
          hidden: false,
        },
      },
    ],
  },
];
