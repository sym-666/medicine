<template>
    <!-- 这太过于简洁了吧？ -->
    <!-- 还有copy.vue是什么意思 -->
    <div class="homeContain">
        <div class="homeMainBg">
            <img class="homeMainImg" src="../../assets/images/i0.png" alt="">
            <div class="homeBtn">申请试用</div>
        </div>
        <div v-for="item in indexImgList" :key="item.id" class="homeImgList">

            <img class="homeImg" :src="item.imgUrl" alt="" />

        </div>
    </div>

</template>

<script setup>
import { useRouter } from 'vue-router';


const indexImgList = [
    {
        id: 1,
        imgUrl: new URL('@/assets/images/i1.png', import.meta.url).href,
    },
    {
        id: 2,
        imgUrl: new URL('@/assets/images/i2.png', import.meta.url).href,
    },
    {
        id: 3,
        imgUrl: new URL('@/assets/images/i3.png', import.meta.url).href,
    },
]

</script>

<style lang="scss" scoped>
.homeContain {
    width: 100%;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: #ffffff;

    .homeMainBg {
        width: 100%;
        min-height: 100vh;

        position: relative; // 添加相对定位，作为 .homeBtn 的参考点

        .homeMainImg {
            width: 100%;
            display: block;
        }

        .homeBtn {
            position: absolute; // 绝对定位
            top: 60%; // 调整按钮位置
            left: 50%;
            transform: translate(-50%, -50%); // 让按钮居中
            width: 200px;
            height: 60px;
            background-color: #7fceef;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            cursor: pointer;

            &:hover {
                background-color: #52f3ff; // 悬浮时变亮
            }
        }
    }

    .homeImgList {
        width: 80%;
        height: 80vh;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;

        &:hover {
            transform: scale(1.05); // 悬浮时放大
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); // 添加阴影
        }

        .homeImg {
            width: 100%;
            height: 100%;
            //   border-radius: 20px; // 设置圆角
            transition: border-radius 0.3s ease-in-out;

            &:hover {
                border-radius: 30px; // 悬浮时增加圆角
            }
        }
    }
}
</style>
