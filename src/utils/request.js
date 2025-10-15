import axios from "axios"
import { ElMessage } from "element-plus"
// 1.axios对象的create方法，创建一个axios实例
const request = axios.create({
  // 2.配置对象
  // 基础路径，发请求的时候，路径当中会出现api
  baseURL: "http://127.0.0.1:5173",
  // 请求超时时间
  timeout: 50000,
})


// 2.request实例添加请求与响应拦截器
request.interceptors.request.use(
  (config) => {
    // 拦截到请求配置对象，进行修改
    // 1.config配置对象
    // 2.比如config要携带token，登录请求必须携带token
    // 3.进度条开始
    // 4.config配置对象
    return config
  },
  (err) => {
    // 错误提示
    returnPromise.reject(err)
  }
)

// 响应拦截器
request.interceptors.response.use((response) => {
    // 成功回调
    return response.data
}), (error) => {
    // 失败回调
    // 处理http网络错误
    // 定义变量:存储错误信息
    let msg = ''
    let status = error.response ? error.response.data : null
    switch (status) {
        case 401:
            msg = 'TOKEN过期'
            break;
        case 403:
            msg = '无权访问'
            break;
        case 404:
            msg = '请求地址错误'
            break;
        case 500:
            msg = '服务器出现问题'
            break;
        default:
            msg = '网络出现问题'
            break;
    }
    ElMessage.error(msg || error)
    return Promise.reject(msg || error)
}

// 4.对外暴露
export default request
