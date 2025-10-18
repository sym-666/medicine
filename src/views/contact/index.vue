<template>
    <!-- 联系我们页面 -->
    <div class="contactContain">
        <div class="contactCard">
            <div class="contactCardTop">
                <div class="contactInfo">
                    <div class="contactinfoTitle">
                        联系人
                    </div>
                    <div class="contactinfoText">袁立沛</div>
                </div>
                <div class="contactInfo">
                    <div class="contactinfoTitle">
                        电话/微信
                    </div>
                    <div class="contactinfoText">18612506617</div>
                </div>
                <div class="contactInfo">
                    <div class="contactinfoTitle">
                        邮箱
                    </div>
                    <div class="contactinfoText">yuanlp2004@163.com</div>
                </div>

            </div>
            <div class="contactCardDown">
                <div class="contactDownTitle">药物预测分析平台(PharmaPredict)</div>
                <div class="contactDownText1">A Cutting-Edge Platform for Drug Prediction and Analysis.</div>
                <div class="contactDownText2">AI成药性预测，让你的新药设计更高效</div>
            </div>

        </div>

    </div>




</template>

<script setup>



</script>


<style lang="scss" scoped>
.contactContain {
    width: 100%;
    height: 120vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background-image: linear-gradient(to top, #a8edea 0%, #fed6e3 100%);

    .contactCard {
        width: 80%;
        height: 400px;
        margin-top: -200px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        display: flex;
        flex-direction: column;

        .contactCardTop {
            width: 100%;
            height: 50%;
            background-color: transparent;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(168, 237, 234, 0.2);

            .contactInfo {
                width: 30%;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;

                .contactinfoTitle {
                    width: 100%;
                    height: 30%;
                    color: gray;

                    .contactimg {
                        width: 30px;
                        height: 30px;
                    }
                }

                .contactinfoText {
                    font-size: 30px;
                    font-weight: 800;
                }
            }
        }

        .contactCardDown {
            width: 100%;
            height: 50%;
            background-color: #ffffff;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 20px;

            .contactDownTitle {
                font-size: 30px;
                /* 字体大小 */
                font-weight: 700;
                /* 字体粗细 */
                margin-bottom: 10px;
                /* 与下方内容的间距 */
            }

            .contactDownText1 {
                color: #808080;
                /* 字体颜色为灰色 */
                font-size: 16px;
                /* 字体大小 */
                margin-bottom: 10px;
                /* 与下方内容的间距 */
            }

            .contactDownText2 {
                font-size: 16px;
                /* 字体大小 */
                color: #333;
                /* 字体颜色 */
            }
        }
    }
}
</style>
